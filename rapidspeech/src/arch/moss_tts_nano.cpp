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
  std::vector<int> prompt17;                 // 17-wide prompt, flattened [L*17] (tokenizer stage populates)
  std::vector<std::vector<int>> gen_frames;  // generated codebook frames [N][16]
  std::vector<float> audio_out;              // produced waveform (streaming drain)
  size_t audio_drained = 0;
  int max_frames = 750;                      // ~60s cap
  bool do_sample = true;
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

// --- generation core (verified in tools/moss_gen.cpp; graphs built on backend_) ---
#include "ggml-cpu.h"
#include <cmath>
#include <algorithm>

bool MossTTSNanoModel::AllocKV() {
  const int HD = hp_.head_dim, NH = hp_.n_head, NL = hp_.n_layer;
  ggml_init_params kp{ (size_t)(2 * NL + 4) * ggml_tensor_overhead(), nullptr, true };
  kv_ctx_ = ggml_init(kp);
  kv_k_.resize(NL); kv_v_.resize(NL);
  for (int i = 0; i < NL; i++) {
    kv_k_[i] = ggml_new_tensor_3d(kv_ctx_, GGML_TYPE_F32, HD, NH, kMaxSeq);
    kv_v_[i] = ggml_new_tensor_3d(kv_ctx_, GGML_TYPE_F32, HD, NH, kMaxSeq);
  }
  return ggml_backend_alloc_ctx_tensors(kv_ctx_, backend_) != nullptr;
}

static ggml_tensor *ln(ggml_context *c, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b, float eps) {
  return ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), w), b);
}

// global transformer with KV cache; returns [H, n_new] host floats
std::vector<float> MossTTSNanoModel::RunGlobal(const std::vector<float> &emb, int n_new, int n_past) {
  const int H = hp_.n_embd, NH = hp_.n_head, HD = hp_.head_dim, NL = hp_.n_layer, n_kv = n_past + n_new;
  const float eps = 1e-5f, rb = hp_.rope_base;
  ggml_init_params cp{ (size_t)256 * 1024 * 1024, nullptr, true };
  ggml_context *ctx = ggml_init(cp);
  ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, n_new); ggml_set_input(inp);
  ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new); ggml_set_input(pos);
  ggml_tensor *h = inp; ggml_cgraph *gf = ggml_new_graph(ctx);
  for (int i = 0; i < NL; i++) {
    MossBlock &bl = blocks_[i];
    ggml_tensor *a = ln(ctx, h, bl.attn_norm_w, bl.attn_norm_b, eps);
    ggml_tensor *qkv = ggml_add(ctx, ggml_mul_mat(ctx, bl.attn_qkv_w, a), bl.attn_qkv_b);
    ggml_tensor *q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, n_new, qkv->nb[1], 0)), HD, NH, n_new);
    ggml_tensor *k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, n_new, qkv->nb[1], H * 4)), HD, NH, n_new);
    ggml_tensor *v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, n_new, qkv->nb[1], 2 * H * 4)), HD, NH, n_new);
    q = ggml_rope_ext(ctx, q, pos, nullptr, HD, 0, 0, rb, 1, 0, 1, 0, 0);
    k = ggml_rope_ext(ctx, k, pos, nullptr, HD, 0, 0, rb, 1, 0, 1, 0, 0);
    ggml_tensor *kd = ggml_view_3d(ctx, kv_k_[i], HD, NH, n_new, kv_k_[i]->nb[1], kv_k_[i]->nb[2], (size_t)n_past * kv_k_[i]->nb[2]);
    ggml_tensor *vd = ggml_view_3d(ctx, kv_v_[i], HD, NH, n_new, kv_v_[i]->nb[1], kv_v_[i]->nb[2], (size_t)n_past * kv_v_[i]->nb[2]);
    ggml_build_forward_expand(gf, ggml_cpy(ctx, k, kd));
    ggml_build_forward_expand(gf, ggml_cpy(ctx, v, vd));
    ggml_tensor *Ka = ggml_view_3d(ctx, kv_k_[i], HD, NH, n_kv, kv_k_[i]->nb[1], kv_k_[i]->nb[2], 0);
    ggml_tensor *Va = ggml_view_3d(ctx, kv_v_[i], HD, NH, n_kv, kv_v_[i]->nb[1], kv_v_[i]->nb[2], 0);
    ggml_tensor *kq = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, ggml_permute(ctx, Ka, 0, 2, 1, 3), ggml_permute(ctx, q, 0, 2, 1, 3)), 1.0f / sqrtf(HD)), n_past));
    ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, Va, 1, 2, 0, 3));
    ggml_tensor *kqv = ggml_permute(ctx, ggml_mul_mat(ctx, vt, kq), 0, 2, 1, 3);
    ggml_tensor *att = ggml_add(ctx, ggml_mul_mat(ctx, bl.attn_out_w, ggml_cont_2d(ctx, kqv, H, n_new)), bl.attn_out_b);
    h = ggml_add(ctx, h, att);
    ggml_tensor *f = ln(ctx, h, bl.ffn_norm_w, bl.ffn_norm_b, eps);
    ggml_tensor *up = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, bl.ffn_up_w, f), bl.ffn_up_b));
    h = ggml_add(ctx, h, ggml_add(ctx, ggml_mul_mat(ctx, bl.ffn_down_w, up), bl.ffn_down_b));
  }
  h = ln(ctx, h, output_norm_w_, output_norm_b_, eps); ggml_set_output(h);
  ggml_build_forward_expand(gf, h);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  ggml_gallocr_alloc_graph(ga, gf);
  ggml_backend_tensor_set(inp, emb.data(), 0, (size_t)H * n_new * 4);
  std::vector<int32_t> pv(n_new); for (int j = 0; j < n_new; j++) pv[j] = n_past + j;
  ggml_backend_tensor_set(pos, pv.data(), 0, n_new * 4);
  ggml_backend_graph_compute(backend_, gf);
  std::vector<float> out((size_t)H * n_new); ggml_backend_tensor_get(h, out.data(), 0, out.size() * 4);
  ggml_gallocr_free(ga); ggml_free(ctx);
  return out;
}

// 1-layer local transformer over L positions; returns last-position hidden [H]
std::vector<float> MossTTSNanoModel::RunLocal(const std::vector<float> &seq, int L) {
  const int H = hp_.n_embd, NH = hp_.n_head, HD = hp_.head_dim;
  const float eps = 1e-5f, rb = hp_.rope_base;
  ggml_init_params cp{ (size_t)64 * 1024 * 1024, nullptr, true };
  ggml_context *ctx = ggml_init(cp);
  ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, H, L); ggml_set_input(inp);
  ggml_tensor *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L); ggml_set_input(pos);
  MossBlock &bl = local_blocks_[0]; ggml_tensor *h = inp;
  ggml_tensor *a = ln(ctx, h, bl.attn_norm_w, bl.attn_norm_b, eps);
  ggml_tensor *qkv = ggml_add(ctx, ggml_mul_mat(ctx, bl.attn_qkv_w, a), bl.attn_qkv_b);
  ggml_tensor *q = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, L, qkv->nb[1], 0)), HD, NH, L);
  ggml_tensor *k = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, L, qkv->nb[1], H * 4)), HD, NH, L);
  ggml_tensor *v = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, qkv, H, L, qkv->nb[1], 2 * H * 4)), HD, NH, L);
  q = ggml_rope_ext(ctx, q, pos, nullptr, HD, 0, 0, rb, 1, 0, 1, 0, 0);
  k = ggml_rope_ext(ctx, k, pos, nullptr, HD, 0, 0, rb, 1, 0, 1, 0, 0);
  ggml_tensor *kq = ggml_soft_max(ctx, ggml_diag_mask_inf(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), ggml_permute(ctx, q, 0, 2, 1, 3)), 1.0f / sqrtf(HD)), 0));
  ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
  ggml_tensor *kqv = ggml_permute(ctx, ggml_mul_mat(ctx, vt, kq), 0, 2, 1, 3);
  ggml_tensor *att = ggml_add(ctx, ggml_mul_mat(ctx, bl.attn_out_w, ggml_cont_2d(ctx, kqv, H, L)), bl.attn_out_b);
  h = ggml_add(ctx, h, att);
  ggml_tensor *f = ln(ctx, h, bl.ffn_norm_w, bl.ffn_norm_b, eps);
  ggml_tensor *up = ggml_gelu(ctx, ggml_add(ctx, ggml_mul_mat(ctx, bl.ffn_up_w, f), bl.ffn_up_b));
  h = ggml_add(ctx, h, ggml_add(ctx, ggml_mul_mat(ctx, bl.ffn_down_w, up), bl.ffn_down_b));
  h = ln(ctx, h, local_out_norm_w_, local_out_norm_b_, eps); ggml_set_output(h);
  ggml_cgraph *gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, h);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
  ggml_gallocr_alloc_graph(ga, gf);
  ggml_backend_tensor_set(inp, seq.data(), 0, (size_t)H * L * 4);
  std::vector<int32_t> pv(L); for (int j = 0; j < L; j++) pv[j] = j;
  ggml_backend_tensor_set(pos, pv.data(), 0, L * 4);
  ggml_backend_graph_compute(backend_, gf);
  std::vector<float> full((size_t)H * L); ggml_backend_tensor_get(h, full.data(), 0, full.size() * 4);
  ggml_gallocr_free(ga); ggml_free(ctx);
  return std::vector<float>(full.begin() + (size_t)(L - 1) * H, full.end());
}

// embedding row lookup (Q8 tensor -> host [H]) via get_rows
static std::vector<float> emb_row(ggml_backend_t be, ggml_tensor *tbl, int id, int H) {
  ggml_init_params cp{ (size_t)8 * 1024 * 1024, nullptr, true }; ggml_context *ctx = ggml_init(cp);
  ggml_tensor *idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1); ggml_set_input(idx);
  ggml_tensor *r = ggml_get_rows(ctx, tbl, idx); ggml_set_output(r);
  ggml_cgraph *gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, r);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
  ggml_gallocr_alloc_graph(ga, gf);
  ggml_backend_tensor_set(idx, &id, 0, 4);
  ggml_backend_graph_compute(be, gf);
  std::vector<float> out(H); ggml_backend_tensor_get(r, out.data(), 0, H * 4);
  ggml_gallocr_free(ga); ggml_free(ctx); return out;
}
// logits = tbl(rows,H) . h ; sample (greedy or temp/top-k)
static int sample_head(ggml_backend_t be, ggml_tensor *tbl, const std::vector<float> &h, int rows, int H,
                       bool do_sample, float temp, int topk, uint32_t &rng) {
  ggml_init_params cp{ (size_t)32 * 1024 * 1024, nullptr, true }; ggml_context *ctx = ggml_init(cp);
  ggml_tensor *hv = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H); ggml_set_input(hv);
  ggml_tensor *lg = ggml_mul_mat(ctx, tbl, hv); ggml_set_output(lg);
  ggml_cgraph *gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, lg);
  ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
  ggml_gallocr_alloc_graph(ga, gf);
  ggml_backend_tensor_set(hv, h.data(), 0, H * 4);
  ggml_backend_graph_compute(be, gf);
  std::vector<float> l(rows); ggml_backend_tensor_get(lg, l.data(), 0, rows * 4);
  ggml_gallocr_free(ga); ggml_free(ctx);
  if (!do_sample) { int b = 0; for (int r = 1; r < rows; r++) if (l[r] > l[b]) b = r; return b; }
  std::vector<int> idx(rows); for (int r = 0; r < rows; r++) idx[r] = r;
  if (topk > 0 && topk < rows) { std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b){ return l[a] > l[b]; }); idx.resize(topk); }
  float mx = -1e30f; for (int i : idx) mx = std::max(mx, l[i]);
  double sum = 0; std::vector<double> pr(idx.size());
  for (size_t i = 0; i < idx.size(); i++) { pr[i] = exp((l[idx[i]] - mx) / temp); sum += pr[i]; }
  rng = rng * 1664525u + 1013904223u; double u = (double)(rng >> 8) / 16777216.0 * sum, c = 0;
  for (size_t i = 0; i < idx.size(); i++) { c += pr[i]; if (u <= c) return idx[i]; }
  return idx.back();
}

// one frame: local greedy/sampled over 16 codebooks with real feedback
std::vector<int> MossTTSNanoModel::GenFrame(const std::vector<float> &hidden, uint32_t &rng, bool do_sample) {
  const int H = hp_.n_embd, CB = hp_.codebook_size, NCB = hp_.n_codebooks;
  std::vector<float> lseq(hidden); int Lc = 1;
  std::vector<float> hl = RunLocal(lseq, Lc);
  int text_tok = sample_head(backend_, token_embd_, hl, hp_.vocab_size, H, false, 1, 0, rng);
  std::vector<int> frame;
  if (text_tok == hp_.audio_end) return frame;               // stop signal
  std::vector<float> cur = emb_row(backend_, token_embd_, text_tok, H);
  for (int k = 0; k < NCB; k++) {
    lseq.insert(lseq.end(), cur.begin(), cur.end()); Lc++;
    hl = RunLocal(lseq, Lc);
    int cb = sample_head(backend_, audio_embd_[k], hl, CB, H, do_sample, 0.8f, 25, rng);
    frame.push_back(cb);
    cur = emb_row(backend_, audio_embd_[k], cb, H);
  }
  return frame;
}

// --- interface ---
bool MossTTSNanoModel::PushText(RSState &state, const char *text, const char *, const char *) {
  auto &s = static_cast<MossState &>(state);
  // NOTE: tokenizer (sentencepiece) + voice-clone prompt build pending; expects
  // s.text_tokens pre-populated (17-wide prompt) by the caller/tokenizer stage.
  if (s.text_tokens.empty()) { RS_LOG_WARN("moss: PushText needs pre-tokenized prompt (tokenizer stage pending)"); return false; }
  return true;
}
bool MossTTSNanoModel::PushReferenceAudio(RSState &, const float *, int, int, ggml_backend_sched_t) {
  return false;  // encode via codec.cpp offline for now
}
bool MossTTSNanoModel::PushReferenceText(RSState &state, const char *ref_text) {
  static_cast<MossState &>(state).ref_text = ref_text ? ref_text : "";
  return true;
}
// 17-wide prompt -> [H, L] embedding = token_embd[col0] + Sum_k audio_embd_k[col_{k+1}] (pad -> 0)
std::vector<float> MossTTSNanoModel::PromptEmbed(const std::vector<int> &p17, int L) {
  const int H = hp_.n_embd, NCB = hp_.n_codebooks;
  std::vector<float> emb((size_t)H * L, 0.0f);
  for (int p = 0; p < L; p++) {
    int base = p * (NCB + 1);
    std::vector<float> t = emb_row(backend_, token_embd_, p17[base], H);
    for (int j = 0; j < H; j++) emb[(size_t)p * H + j] = t[j];
    for (int k = 0; k < NCB; k++) {
      int c = p17[base + 1 + k];
      if (c == hp_.audio_pad || c >= (int)hp_.codebook_size) continue;
      std::vector<float> a = emb_row(backend_, audio_embd_[k], c, H);
      for (int j = 0; j < H; j++) emb[(size_t)p * H + j] += a[j];
    }
  }
  return emb;
}

// Decode: prefill prompt -> AR frame loop (GenFrame + decode_step feedback) -> gen_frames.
// Codec decode (frames -> 48kHz) is the codec.cpp bridge (frames handed off / decoded separately).
bool MossTTSNanoModel::Decode(RSState &state, ggml_backend_sched_t) {
  auto &s = static_cast<MossState &>(state);
  if (!kv_ctx_ && !AllocKV()) { RS_LOG_ERR("moss: KV alloc failed"); return false; }
  if (s.prompt17.empty()) { RS_LOG_ERR("moss: no prompt (tokenizer stage must populate prompt17)"); return false; }
  const int H = hp_.n_embd, NCB = hp_.n_codebooks;
  int L = (int)s.prompt17.size() / (NCB + 1);
  uint32_t rng = (uint32_t)(seed_ ? seed_ : 1234u);
  // prefill
  std::vector<float> pe = PromptEmbed(s.prompt17, L);
  std::vector<float> gh = RunGlobal(pe, L, 0);
  std::vector<float> hidden(gh.begin() + (size_t)(L - 1) * H, gh.end());
  int n_past = L;
  s.gen_frames.clear();
  for (int fi = 0; fi < s.max_frames; fi++) {
    std::vector<int> frame = GenFrame(hidden, rng, s.do_sample);
    if ((int)frame.size() < NCB) break;               // end token
    s.gen_frames.push_back(frame);
    // feedback: next 17-wide row embed = wte(assistant_slot) + Sum audio_embd_k(code_k)
    std::vector<float> row = emb_row(backend_, token_embd_, hp_.audio_assistant_slot, H);
    for (int k = 0; k < NCB; k++) {
      std::vector<float> a = emb_row(backend_, audio_embd_[k], frame[k], H);
      for (int j = 0; j < H; j++) row[j] += a[j];
    }
    std::vector<float> nh = RunGlobal(row, 1, n_past);
    hidden.assign(nh.begin(), nh.begin() + H);
    n_past++;
    if (n_past >= kMaxSeq - 1) break;
  }
  RS_LOG_INFO("moss: generated %zu frames (%.1fs). Codec decode via codec.cpp bridge.",
              s.gen_frames.size(), s.gen_frames.size() / 12.5);
  return true;
}
int MossTTSNanoModel::GetAudioOutput(RSState &state, float **out) {
  auto &s = static_cast<MossState &>(state);
  if (s.audio_drained >= s.audio_out.size()) return 0;
  *out = s.audio_out.data() + s.audio_drained;
  int n = (int)(s.audio_out.size() - s.audio_drained);
  s.audio_drained = s.audio_out.size();
  return n;
}

// --- registration ---
static bool s_moss_reg = [] {
  rs_register_model_arch("moss_tts_nano",
                         []() { return std::make_shared<MossTTSNanoModel>(); });
  return true;
}();
