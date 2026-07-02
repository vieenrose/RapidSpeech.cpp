// Parity harness for the mbistft-vits arch (PrimeTTS v2 / MB-iSTFT-VITS).
//
// Loads the gguf, reads the fixed (phone,tone,lang) id streams from
// parity_inputs.json, runs the full pipeline per utterance with
// noise_scale=0 / length_scale=1 and dumps the 18 reference intermediates
// (emb, attn0..5, enc, m_p, logs_p, logw, w_ceil, m_p_exp, logs_p_exp,
// z_p, z_flow, o_mb, wav) as raw f32 files in PyTorch memory order.
// Compare against /home/luigi/mbvits_run/parity_refs with
// tools/mbistft_parity_compare.py (gate: cosine > 0.99 per module).
//
// Usage: mbistft-parity <model.gguf> <parity_inputs.json> <outdir>
//                       [--refs <parity_refs_dir>] [--rtf]
// --refs teacher-forces the reference w_ceil durations for the downstream
// modules (guards against knife-edge ceil(exp(logw)) off-by-one frames);
// the model's own w_ceil prediction is still dumped and gated.
#include "rapidspeech.h"
#include "core/rs_context.h"
#include "arch/mbistft_vits.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// --- minimal JSON int-array extraction (exact-key match) ---
static bool extract_int_array(const std::string& chunk, const std::string& key,
                              std::vector<int32_t>& out) {
  const std::string pat = "\"" + key + "\":";
  size_t p = chunk.find(pat);
  if (p == std::string::npos) return false;
  p = chunk.find('[', p);
  if (p == std::string::npos) return false;
  size_t e = chunk.find(']', p);
  if (e == std::string::npos) return false;
  out.clear();
  const char* c = chunk.c_str() + p + 1;
  const char* end = chunk.c_str() + e;
  while (c < end) {
    while (c < end && (*c == ',' || *c == ' ' || *c == '\n' || *c == '\r' || *c == '\t')) c++;
    if (c >= end) break;
    out.push_back((int32_t)strtol(c, const_cast<char**>(&c), 10));
  }
  return true;
}

struct Utt {
  int idx;
  std::vector<int32_t> phone_ids, tone_ids, lang_ids;
};

static bool load_inputs(const char* path, std::vector<Utt>& utts) {
  std::ifstream f(path);
  if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
  std::stringstream ss; ss << f.rdbuf();
  std::string s = ss.str();

  // rows are objects each starting with "idx": N
  size_t pos = 0;
  while (true) {
    size_t p = s.find("\"idx\":", pos);
    if (p == std::string::npos) break;
    size_t next = s.find("\"idx\":", p + 6);
    std::string chunk = s.substr(p, (next == std::string::npos ? s.size() : next) - p);
    Utt u;
    u.idx = (int)strtol(chunk.c_str() + 6, nullptr, 10);
    // NOTE: exact keys — "phone_ids" also prefixes "phone_ids_raw", but the
    // pattern includes the trailing ':' so "phone_ids_raw" cannot match.
    if (!extract_int_array(chunk, "phone_ids", u.phone_ids) ||
        !extract_int_array(chunk, "tone_ids", u.tone_ids) ||
        !extract_int_array(chunk, "lang_ids", u.lang_ids)) {
      fprintf(stderr, "utt %d: missing id arrays\n", u.idx);
      return false;
    }
    utts.push_back(std::move(u));
    pos = p + 6;
  }
  return !utts.empty();
}

// Minimal .npy loader for little-endian f32 C-order arrays (returns flat data).
static bool load_npy_f32(const std::string& path, std::vector<float>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[6];
  f.read(magic, 6);
  if (memcmp(magic, "\x93NUMPY", 6) != 0) return false;
  unsigned char ver[2];
  f.read((char*)ver, 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16; f.read((char*)&h16, 2); hlen = h16;
  } else {
    f.read((char*)&hlen, 4);
  }
  std::string hdr(hlen, '\0');
  f.read(&hdr[0], hlen);
  if (hdr.find("'<f4'") == std::string::npos) return false;
  out.clear();
  float buf[4096];
  while (f) {
    f.read((char*)buf, sizeof(buf));
    std::streamsize got = f.gcount();
    out.insert(out.end(), buf, buf + got / 4);
  }
  return true;
}

static void write_raw(const std::string& path, const std::vector<float>& v) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) { fprintf(stderr, "cannot write %s\n", path.c_str()); exit(1); }
  fwrite(v.data(), sizeof(float), v.size(), f);
  fclose(f);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    printf("usage: mbistft-parity <model.gguf> <parity_inputs.json> <outdir> [--rtf]\n");
    return 1;
  }
  const char* gguf_path = argv[1];
  const char* inputs_path = argv[2];
  const std::string outdir = argv[3];
  bool do_rtf = false;
  std::string refs_dir;
  for (int i = 4; i < argc; i++) {
    if (strcmp(argv[i], "--rtf") == 0) do_rtf = true;
    else if (strcmp(argv[i], "--refs") == 0 && i + 1 < argc) refs_dir = argv[++i];
  }

  std::vector<Utt> utts;
  if (!load_inputs(inputs_path, utts)) return 1;
  printf("[inputs] %zu utterances\n", utts.size());

  rs_init_params_t p = rs_default_params();
  p.model_path = gguf_path;
  if (getenv("MBISTFT_CPU")) p.use_gpu = false;  // force CPU-only (watchdog-safe correctness/RTF baseline)
  if (const char* nt = getenv("MBISTFT_THREADS")) {  // override CPU thread count (default 4)
    int n = atoi(nt);
    if (n >= 1) p.n_threads = n;
  }
  printf("[cfg] use_gpu=%d n_threads=%d\n", p.use_gpu, p.n_threads);
  rs_context_t* ctx = rs_init_from_file(p);
  if (!ctx || !ctx->model) { fprintf(stderr, "model load failed\n"); return 1; }

  // ---- parity dump pass ----
  for (auto& u : utts) {
    auto state = ctx->model->CreateState();
    auto* s = static_cast<MBIstftVitsState*>(state.get());
    s->phone_ids = u.phone_ids;
    s->tone_ids = u.tone_ids;
    s->lang_ids = u.lang_ids;
    s->noise_scale = 0.0f;
    s->length_scale = 1.0f;
    s->capture = true;

    // Teacher-force the reference durations for the downstream modules so a
    // knife-edge ceil(exp(logw)) boundary (w within float-eps of an integer)
    // can't desync the frame axis.  The model's own w_ceil prediction is
    // still dumped and compared as its own module.
    if (!refs_dir.empty()) {
      char rp[512];
      snprintf(rp, sizeof(rp), "%s/utt%02d_w_ceil.npy", refs_dir.c_str(), u.idx);
      std::vector<float> wc;
      if (load_npy_f32(rp, wc) && wc.size() == u.phone_ids.size()) {
        s->duration_override.resize(wc.size());
        for (size_t i = 0; i < wc.size(); i++)
          s->duration_override[i] = (int32_t)lrintf(wc[i]);
      } else {
        fprintf(stderr, "utt%02d: could not load %s\n", u.idx, rp);
      }
    }

    if (!ctx->model->Encode({}, *state, ctx->sched)) {
      fprintf(stderr, "utt%02d: Encode failed\n", u.idx);
      return 1;
    }
    if (!ctx->model->Decode(*state, ctx->sched)) {
      fprintf(stderr, "utt%02d: Decode failed\n", u.idx);
      return 1;
    }

    char pre[64];
    snprintf(pre, sizeof(pre), "utt%02d_", u.idx);
    for (auto& [name, data] : s->dumps) {
      write_raw(outdir + "/" + pre + name + ".bin", data);
    }
    printf("[utt%02d] T_text=%d T_frames=%d wav=%zu dumped %zu modules\n",
           u.idx, s->T_text, s->T_frames, s->audio_output.size(), s->dumps.size());
  }

  // ---- RTF pass (capture off, warm) ----
  if (do_rtf) {
    const int sr = ctx->model->GetMeta().audio_sample_rate;
    // warmup
    {
      auto state = ctx->model->CreateState();
      auto* s = static_cast<MBIstftVitsState*>(state.get());
      s->phone_ids = utts[0].phone_ids;
      s->tone_ids = utts[0].tone_ids;
      s->lang_ids = utts[0].lang_ids;
      s->noise_scale = 0.0f;
      ctx->model->Encode({}, *state, ctx->sched);
      ctx->model->Decode(*state, ctx->sched);
    }
    double total_wall = 0.0, total_audio = 0.0;
    for (auto& u : utts) {
      auto state = ctx->model->CreateState();
      auto* s = static_cast<MBIstftVitsState*>(state.get());
      s->phone_ids = u.phone_ids;
      s->tone_ids = u.tone_ids;
      s->lang_ids = u.lang_ids;
      s->noise_scale = 0.0f;
      auto t0 = std::chrono::high_resolution_clock::now();
      if (!ctx->model->Encode({}, *state, ctx->sched)) return 1;
      if (!ctx->model->Decode(*state, ctx->sched)) return 1;
      double el = std::chrono::duration<double>(
          std::chrono::high_resolution_clock::now() - t0).count();
      double audio_s = (double)s->audio_output.size() / sr;
      total_wall += el;
      total_audio += audio_s;
      printf("[rtf utt%02d] wall=%.3fs audio=%.3fs rtf=%.4f\n", u.idx, el,
             audio_s, el / audio_s);
    }
    printf("[rtf] total wall=%.3fs audio=%.3fs RTF=%.4f (xRT=%.1f)\n",
           total_wall, total_audio, total_wall / total_audio,
           total_audio / total_wall);
  }

  rs_free(ctx);
  return 0;
}
