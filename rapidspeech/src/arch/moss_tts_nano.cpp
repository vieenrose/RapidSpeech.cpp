// MOSS-TTS-Nano-100M — RapidSpeech.cpp / ggml-CUDA arch.
// Stage 1: model struct + GGUF loading (hparams + all Q8_0/F32 tensors) + registration.
// Graphs (global GPT-2 decode, 16-step local loop, 4-stage codec decode, sampling,
// streaming) land in subsequent stages. Contract matches scripts/convert_moss_*_to_gguf.py.
#include "arch/moss_tts_nano.h"
#include "core/rs_context.h"
#include "utils/rs_log.h"
#include <cstdio>
#include <string>

MossTTSNanoModel::MossTTSNanoModel() { meta_.arch_name = "moss_tts_nano"; }
MossTTSNanoModel::~MossTTSNanoModel() = default;

// --- state ---
struct MossState : public RSState {
  std::vector<int> text_tokens;              // input text (streaming)
  std::vector<std::vector<int>> ref_codes;   // reference audio codes [n_codebooks][T]
  std::string ref_text;
  std::vector<float> audio_out;              // produced waveform (streaming drain)
  size_t audio_drained = 0;
};

std::shared_ptr<RSState> MossTTSNanoModel::CreateState() {
  return std::make_shared<MossState>();
}

// --- helpers ---
static uint32_t hp_u32(const gguf_context *g, const char *k, uint32_t dflt, bool *ok) {
  int64_t key = gguf_find_key(g, k);
  if (key < 0) { RS_LOG_WARN("moss: missing hparam %s", k); if (ok) *ok = false; return dflt; }
  return gguf_get_val_u32(g, key);
}
static float hp_f32(const gguf_context *g, const char *k, float dflt) {
  int64_t key = gguf_find_key(g, k);
  return key < 0 ? dflt : gguf_get_val_f32(g, key);
}

bool MossTTSNanoModel::LoadHParams(const rs_context_t &ctx) {
  const gguf_context *g = ctx.ctx_gguf;
  bool ok = true;
  hp_.n_layer       = hp_u32(g, "moss.n_layer", 0, &ok);
  hp_.n_local_layer = hp_u32(g, "moss.n_local_layer", 1, &ok);
  hp_.n_embd        = hp_u32(g, "moss.n_embd", 0, &ok);
  hp_.n_head        = hp_u32(g, "moss.n_head", 0, &ok);
  hp_.n_ff          = hp_u32(g, "moss.n_ff", 0, &ok);
  hp_.head_dim      = hp_u32(g, "moss.head_dim", 0, &ok);
  hp_.vocab_size    = hp_u32(g, "moss.vocab_size", 0, &ok);
  hp_.n_codebooks   = hp_u32(g, "moss.n_codebooks", 0, &ok);
  hp_.codebook_size = hp_u32(g, "moss.codebook_size", 0, &ok);
  hp_.sample_rate   = hp_u32(g, "moss.sample_rate", 48000, &ok);
  hp_.rope_base     = hp_f32(g, "moss.rope_base", 10000.0f);
  hp_.audio_start           = (int)hp_u32(g, "moss.audio_start_token_id", 6, &ok);
  hp_.audio_end             = (int)hp_u32(g, "moss.audio_end_token_id", 7, &ok);
  hp_.audio_pad             = (int)hp_u32(g, "moss.audio_pad_token_id", 1024, &ok);
  hp_.audio_user_slot       = (int)hp_u32(g, "moss.audio_user_slot_token_id", 8, &ok);
  hp_.audio_assistant_slot  = (int)hp_u32(g, "moss.audio_assistant_slot_token_id", 9, &ok);
  meta_.arch_name = "moss_tts_nano";
  meta_.audio_sample_rate = (int)hp_.sample_rate;
  meta_.vocab_size = (int)hp_.vocab_size;
  return ok;
}

static ggml_tensor *need(ggml_context *d, const std::string &name, bool *ok) {
  ggml_tensor *t = ggml_get_tensor(d, name.c_str());
  if (!t) { RS_LOG_WARN("moss: missing tensor %s", name.c_str()); if (ok) *ok = false; }
  return t;
}

static void load_block(ggml_context *d, const std::string &pfx, int i, MossBlock &b, bool *ok) {
  std::string p = pfx + "blk." + std::to_string(i) + ".";
  b.attn_norm_w = need(d, p + "attn_norm.weight", ok);
  b.attn_norm_b = need(d, p + "attn_norm.bias", ok);
  b.attn_qkv_w  = need(d, p + "attn_qkv.weight", ok);
  b.attn_qkv_b  = need(d, p + "attn_qkv.bias", ok);
  b.attn_out_w  = need(d, p + "attn_out.weight", ok);
  b.attn_out_b  = need(d, p + "attn_out.bias", ok);
  b.ffn_norm_w  = need(d, p + "ffn_norm.weight", ok);
  b.ffn_norm_b  = need(d, p + "ffn_norm.bias", ok);
  b.ffn_up_w    = need(d, p + "ffn_up.weight", ok);
  b.ffn_up_b    = need(d, p + "ffn_up.bias", ok);
  b.ffn_down_w  = need(d, p + "ffn_down.weight", ok);
  b.ffn_down_b  = need(d, p + "ffn_down.bias", ok);
}

bool MossTTSNanoModel::LoadTensors(const rs_context_t &ctx) {
  ggml_context *d = ctx.gguf_data;
  bool ok = true;
  token_embd_   = need(d, "token_embd.weight", &ok);
  output_norm_w_ = need(d, "output_norm.weight", &ok);
  output_norm_b_ = need(d, "output_norm.bias", &ok);
  blocks_.resize(hp_.n_layer);
  for (uint32_t i = 0; i < hp_.n_layer; i++) load_block(d, "", i, blocks_[i], &ok);
  local_blocks_.resize(hp_.n_local_layer);
  for (uint32_t i = 0; i < hp_.n_local_layer; i++) load_block(d, "local.", i, local_blocks_[i], &ok);
  local_out_norm_w_ = need(d, "local.output_norm.weight", &ok);
  local_out_norm_b_ = need(d, "local.output_norm.bias", &ok);
  audio_embd_.resize(hp_.n_codebooks);
  for (uint32_t k = 0; k < hp_.n_codebooks; k++)
    audio_embd_[k] = need(d, "audio_embd." + std::to_string(k) + ".weight", &ok);
  return ok;
}

bool MossTTSNanoModel::LoadCodec(const rs_context_t &ctx) {
  ggml_context *d = ctx.gguf_data;
  // codec is optional (may be a separate gguf); probe one tensor first
  if (!ggml_get_tensor(d, "codec.quantizer.output_proj.weight")) {
    RS_LOG_WARN("moss: codec tensors absent (token-only mode)");
    return false;
  }
  bool ok = true;
  codec_q_in_w_  = need(d, "codec.quantizer.input_proj.weight", &ok);
  codec_q_in_b_  = need(d, "codec.quantizer.input_proj.bias", &ok);
  codec_q_out_w_ = need(d, "codec.quantizer.output_proj.weight", &ok);
  codec_q_out_b_ = need(d, "codec.quantizer.output_proj.bias", &ok);
  codec_rvq_.resize(hp_.n_codebooks);
  for (uint32_t k = 0; k < hp_.n_codebooks; k++) {
    std::string p = "codec.quantizer.quantizers." + std::to_string(k) + ".";
    codec_rvq_[k].codebook   = need(d, p + "codebook.weight", &ok);
    codec_rvq_[k].in_proj_w  = need(d, p + "in_proj.weight", &ok);
    codec_rvq_[k].in_proj_b  = need(d, p + "in_proj.bias", &ok);
    codec_rvq_[k].out_proj_w = need(d, p + "out_proj.weight", &ok);
    codec_rvq_[k].out_proj_b = need(d, p + "out_proj.bias", &ok);
  }
  // 4 decoder stages, discover layers per stage dynamically
  for (int s = 0; s < 16; s++) {
    std::string sp = "codec.decoder." + std::to_string(s) + ".";
    if (!ggml_get_tensor(d, (sp + "input_proj.weight").c_str())) continue;  // even idx = param-free upsample
    MossCodecStage st;
    st.input_proj_w  = need(d, sp + "input_proj.weight", &ok);
    st.output_proj_w = need(d, sp + "output_proj.weight", &ok);
    for (int l = 0; l < 64; l++) {
      std::string lp = sp + "transformer.layers." + std::to_string(l) + ".";
      if (!ggml_get_tensor(d, (lp + "self_attn.in_proj.weight").c_str())) break;
      MossCodecLayer cl;
      cl.in_proj_w  = need(d, lp + "self_attn.in_proj.weight", &ok);
      cl.out_proj_w = need(d, lp + "self_attn.out_proj.weight", &ok);
      cl.ffn0_w     = need(d, lp + "ffn.0.weight", &ok);
      cl.ffn1_w     = need(d, lp + "ffn.2.weight", &ok);
      cl.norm1_w    = need(d, lp + "norm1.weight", &ok);
      cl.norm1_b    = need(d, lp + "norm1.bias", &ok);
      cl.norm2_w    = need(d, lp + "norm2.weight", &ok);
      cl.norm2_b    = need(d, lp + "norm2.bias", &ok);
      cl.ls1        = need(d, lp + "layer_scale_1.scale", &ok);
      cl.ls2        = need(d, lp + "layer_scale_2.scale", &ok);
      st.layers.push_back(cl);
    }
    codec_stages_.push_back(std::move(st));
  }
  RS_LOG_INFO("moss codec: %zu stages, %u rvq", codec_stages_.size(), hp_.n_codebooks);
  return ok;
}

bool MossTTSNanoModel::Load(const std::unique_ptr<rs_context_t> &ctx, ggml_backend_t backend) {
  backend_ = backend;
  if (!LoadHParams(*ctx)) { RS_LOG_ERR("moss: hparam load failed"); return false; }
  if (!LoadTensors(*ctx)) { RS_LOG_ERR("moss: AR tensor load failed"); return false; }
  codec_loaded_ = LoadCodec(*ctx);
  RS_LOG_INFO("moss-tts-nano loaded: %u global + %u local layers, %u codebooks, sr=%u, codec=%s",
              hp_.n_layer, hp_.n_local_layer, hp_.n_codebooks, hp_.sample_rate,
              codec_loaded_ ? "yes" : "no");
  return true;
}

// --- generation (stubs — graph stages to follow) ---
bool MossTTSNanoModel::PushText(RSState &state, const char *text, const char *, const char *) {
  auto &s = static_cast<MossState &>(state);
  (void)s; (void)text;
  RS_LOG_WARN("moss: PushText not yet implemented (graph stage pending)");
  return false;
}
bool MossTTSNanoModel::PushReferenceAudio(RSState &, const float *, int, int, ggml_backend_sched_t) {
  return false;  // reference codes are pre-encoded offline for now
}
bool MossTTSNanoModel::PushReferenceText(RSState &state, const char *ref_text) {
  static_cast<MossState &>(state).ref_text = ref_text ? ref_text : "";
  return true;
}
bool MossTTSNanoModel::Decode(RSState &, ggml_backend_sched_t) { return false; }
int MossTTSNanoModel::GetAudioOutput(RSState &, float **) { return 0; }

// --- registration ---
static bool s_moss_reg = [] {
  rs_register_model_arch("moss_tts_nano",
                         []() { return std::make_shared<MossTTSNanoModel>(); });
  return true;
}();
