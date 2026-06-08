#include "cosyvoice3_speech_tokenizer.h"

#include "frontend/cosyvoice3_mel.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

CosyVoice3SpeechTokenizer::~CosyVoice3SpeechTokenizer() = default;

// =====================================================================
// Load — bind hparams + tensors
// =====================================================================

bool CosyVoice3SpeechTokenizer::Load(gguf_context *ctx_gguf,
                                     ggml_context *gguf_data,
                                     ggml_backend_t backend) {
  (void)backend;
  if (!ctx_gguf || !gguf_data) return false;

  if (!LoadHparams(ctx_gguf)) return false;

  std::map<std::string, ggml_tensor *> tensors;
  const int n = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n; ++i) {
    const char *name = gguf_get_tensor_name(ctx_gguf, i);
    ggml_tensor *t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }
  if (!LoadTensors(tensors)) return false;

  loaded_ = true;
  RS_LOG_INFO("CosyVoice3SpeechTokenizer loaded: depth=%d dim=%d heads=%d "
              "head_dim=%d ff=%d fsmn_k=%d pad=%d/%d codebook=%d use_rope=%d",
              depth_, dim_, n_heads_, head_dim_, ff_dim_, fsmn_kernel_,
              fsmn_pad_l_, fsmn_pad_r_, codebook_, (int)use_rope_);
  return true;
}

bool CosyVoice3SpeechTokenizer::LoadHparams(gguf_context *ctx_gguf) {
  auto u32 = [&](const char *k, int dflt) {
    int id = gguf_find_key(ctx_gguf, k);
    return id >= 0 ? (int)gguf_get_val_u32(ctx_gguf, id) : dflt;
  };
  auto f32 = [&](const char *k, float dflt) {
    int id = gguf_find_key(ctx_gguf, k);
    return id >= 0 ? gguf_get_val_f32(ctx_gguf, id) : dflt;
  };
  depth_       = u32("cosyvoice3.tokenizer.depth",         12);
  dim_         = u32("cosyvoice3.tokenizer.dim",           1280);
  n_heads_     = u32("cosyvoice3.tokenizer.n_heads",       20);
  head_dim_    = u32("cosyvoice3.tokenizer.head_dim",      64);
  ff_dim_      = u32("cosyvoice3.tokenizer.ff_dim",        5120);
  fsmn_kernel_ = u32("cosyvoice3.tokenizer.fsmn_kernel",   31);
  fsmn_pad_l_  = u32("cosyvoice3.tokenizer.fsmn_pad_left", 15);
  fsmn_pad_r_  = u32("cosyvoice3.tokenizer.fsmn_pad_right",15);
  codebook_    = u32("cosyvoice3.tokenizer.codebook",      6561);
  proj_dim_    = u32("cosyvoice3.tokenizer.proj_dim",      8);
  n_mels_      = u32("cosyvoice3.tokenizer.n_mels",        128);
  use_rope_    = u32("cosyvoice3.tokenizer.use_rope",      1) != 0;
  rope_theta_  = f32("cosyvoice3.tokenizer.rope_theta",    10000.0f);
  rope_max_pos_= u32("cosyvoice3.tokenizer.rope_max_pos",  4096);
  fsq_gain_    = f32("cosyvoice3.tokenizer.fsq_gain",      0.999f);
  attn_ln_eps_ = f32("cosyvoice3.tokenizer.attn_ln_eps",   1e-6f);
  mlp_ln_eps_  = f32("cosyvoice3.tokenizer.mlp_ln_eps",    1e-5f);
  return true;
}

bool CosyVoice3SpeechTokenizer::LoadTensors(
    const std::map<std::string, ggml_tensor *> &tensors) {
  auto get = [&](const char *name, ggml_tensor **dst, bool required = true) {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      if (required) RS_LOG_ERR("Tokenizer: missing %s", name);
      return !required;
    }
    *dst = it->second;
    return true;
  };
  bool ok = true;
  ok &= get("encoder.conv1.weight", &conv1_w_);
  ok &= get("encoder.conv1.bias",   &conv1_b_);
  ok &= get("encoder.conv2.weight", &conv2_w_);
  ok &= get("encoder.conv2.bias",   &conv2_b_);
  ok &= get("quant.pd.weight",      &pd_w_);
  ok &= get("quant.pd.bias",        &pd_b_);

  blocks_.resize(depth_);
  char buf[128];
  for (int i = 0; i < depth_; ++i) {
    auto &b = blocks_[i];
    auto fmt = [&](const char *t) {
      snprintf(buf, sizeof(buf), "enc.b.%d.%s", i, t); return buf;
    };
    ok &= get(fmt("attn_ln.weight"),     &b.attn_ln_w);
    ok &= get(fmt("attn_ln.bias"),       &b.attn_ln_b);
    ok &= get(fmt("attn.query.weight"),  &b.q_w);
    ok &= get(fmt("attn.query.bias"),    &b.q_b);
    ok &= get(fmt("attn.key.weight"),    &b.k_w);   // no K bias
    ok &= get(fmt("attn.value.weight"),  &b.v_w);
    ok &= get(fmt("attn.value.bias"),    &b.v_b);
    ok &= get(fmt("attn.fsmn.weight"),   &b.fsmn_w);
    ok &= get(fmt("attn.out.weight"),    &b.o_w);
    ok &= get(fmt("attn.out.bias"),      &b.o_b);
    ok &= get(fmt("mlp_ln.weight"),      &b.mlp_ln_w);
    ok &= get(fmt("mlp_ln.bias"),        &b.mlp_ln_b);
    ok &= get(fmt("mlp.0.weight"),       &b.mlp0_w);
    ok &= get(fmt("mlp.0.bias"),         &b.mlp0_b);
    ok &= get(fmt("mlp.2.weight"),       &b.mlp2_w);
    ok &= get(fmt("mlp.2.bias"),         &b.mlp2_b);
  }
  return ok;
}

// =====================================================================
// Encoder graph
// =====================================================================

// Helper: turn a 1-D bias [C] into a (1, C, 1) view for broadcast over (T, C).
static ggml_tensor *bias_1d(ggml_context *ctx, ggml_tensor *b) {
  return ggml_reshape_3d(ctx, b, 1, b->ne[0], 1);
}

ggml_tensor *CosyVoice3SpeechTokenizer::BuildEncoderGraph(
    ggml_context *ctx, ggml_cgraph *gf,
    ggml_tensor *mel_in, int T_mel) const {
  const bool dbg = std::getenv("RS_CV3_TOK_DEBUG") != nullptr;
  auto mark = [&](ggml_tensor *t, const char *name) {
    if (dbg) {
      ggml_set_name(t, name);
      ggml_set_output(t);
      ggml_build_forward_expand(gf, t);
    }
  };

  // mel_in ne = (T_mel, n_mels=128) — channel-first row-major (matches the
  // V2/voxtral conv1d input convention).
  ggml_tensor *x = ggml_conv_1d(ctx, conv1_w_, mel_in, /*s*/ 2, /*p*/ 1, 1);
  x = ggml_add(ctx, x, bias_1d(ctx, conv1_b_));
  x = ggml_gelu(ctx, x);                                      // (T/2, 1280)
  mark(x, "dbg_post_conv1_gelu");
  x = ggml_conv_1d(ctx, conv2_w_, x, /*s*/ 2, /*p*/ 1, 1);
  x = ggml_add(ctx, x, bias_1d(ctx, conv2_b_));
  x = ggml_gelu(ctx, x);                                      // (T/4, 1280)
  mark(x, "dbg_post_conv2_gelu");

  const int T_tok = (int)x->ne[0];
  // Reshape to (T_tok, 1280) then transpose to (1280, T_tok) so the
  // transformer ops use ne[0]=feature_dim (matches V2 pattern).
  x = ggml_reshape_2d(ctx, x, T_tok, dim_);
  x = ggml_cont(ctx, ggml_transpose(ctx, x));                 // (1280, T_tok)
  mark(x, "dbg_transposed");

  // Position ids for RoPE (only used when use_rope_ is true).
  ggml_tensor *positions = nullptr;
  if (use_rope_) {
    positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T_tok);
    ggml_set_input(positions);
    ggml_set_name(positions, "v3tok_positions");
  }

  for (int il = 0; il < depth_; ++il) {
    const auto &b = blocks_[il];
    ggml_tensor *residual = x;

    // attn_ln (eps=1e-6).
    ggml_tensor *h = ggml_norm(ctx, x, attn_ln_eps_);
    h = ggml_mul(ctx, h, b.attn_ln_w);
    h = ggml_add(ctx, h, b.attn_ln_b);
    if (il == 0) mark(h, "dbg_blk0_attn_ln");

    // Q/K/V projections — Q and V biased, K not.
    ggml_tensor *Q = ggml_add(ctx, ggml_mul_mat(ctx, b.q_w, h), b.q_b);
    ggml_tensor *K =          ggml_mul_mat(ctx, b.k_w, h);
    ggml_tensor *V = ggml_add(ctx, ggml_mul_mat(ctx, b.v_w, h), b.v_b);
    if (il == 0) { mark(Q, "dbg_blk0_Q"); mark(K, "dbg_blk0_K"); mark(V, "dbg_blk0_V"); }
    // Q, K, V ne = (1280, T_tok)

    // FSMN side branch on V. Transpose V to (T_tok, 1280), depthwise conv
    // (k=31, sym pad 15/15), add V residual, transpose back.
    ggml_tensor *v_t = ggml_cont(ctx, ggml_transpose(ctx, V));  // (T_tok, 1280)
    GGML_ASSERT(fsmn_pad_l_ == fsmn_pad_r_ &&
                "V3 FSMN is symmetric; asymmetric path not yet implemented");
    ggml_tensor *fsmn = ggml_conv_1d_dw(ctx, b.fsmn_w, v_t,
                                        /*s*/ 1, /*p*/ fsmn_pad_l_, /*d*/ 1);
    if (ggml_n_dims(fsmn) > 2) {
      fsmn = ggml_reshape_2d(ctx, fsmn, fsmn->ne[0], fsmn->ne[1]);
    }
    fsmn = ggml_add(ctx, fsmn, v_t);
    fsmn = ggml_cont(ctx, ggml_transpose(ctx, fsmn));          // (1280, T_tok)
    if (il == 0) mark(fsmn, "dbg_blk0_fsmn_out");

    // Multi-head attention. V3 USES standard NEOX-style RoPE applied to Q
    // and K (cos/sin tables baked into the ONNX as constants, theta=10000,
    // head_dim=64). Reshape to (head_dim, n_heads, T_tok), then apply RoPE
    // before the (0,2,1,3) permute to flash-attn layout.
    Q = ggml_reshape_3d(ctx, Q, head_dim_, n_heads_, T_tok);
    K = ggml_reshape_3d(ctx, K, head_dim_, n_heads_, T_tok);
    V = ggml_reshape_3d(ctx, V, head_dim_, n_heads_, T_tok);
    if (use_rope_ && positions) {
      Q = ggml_rope_ext(ctx, Q, positions, /*freq_factors*/ nullptr,
                        head_dim_, GGML_ROPE_TYPE_NEOX, rope_max_pos_,
                        rope_theta_, /*freq_scale*/ 1.0f, /*ext_factor*/ 0.0f,
                        /*attn_factor*/ 1.0f, /*beta_fast*/ 0.0f,
                        /*beta_slow*/ 0.0f);
      K = ggml_rope_ext(ctx, K, positions, nullptr,
                        head_dim_, GGML_ROPE_TYPE_NEOX, rope_max_pos_,
                        rope_theta_, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }
    Q = ggml_cont(ctx, ggml_permute(ctx, Q, 0, 2, 1, 3));
    K = ggml_cont(ctx, ggml_permute(ctx, K, 0, 2, 1, 3));
    V = ggml_cont(ctx, ggml_permute(ctx, V, 0, 2, 1, 3));

    const float attn_scale = 1.0f / std::sqrt((float)head_dim_);
    ggml_tensor *attn = ggml_flash_attn_ext(
        ctx, Q, K, V, /*mask*/ nullptr, attn_scale,
        /*max_bias*/ 0.0f, /*logit_softcap*/ 0.0f);
    attn = ggml_reshape_2d(ctx, attn, dim_, T_tok);            // (1280, T_tok)

    // Output projection + bias, add FSMN, residual.
    attn = ggml_add(ctx, ggml_mul_mat(ctx, b.o_w, attn), b.o_b);
    attn = ggml_add(ctx, attn, fsmn);
    x = ggml_add(ctx, residual, attn);

    // MLP block.
    residual = x;
    h = ggml_norm(ctx, x, mlp_ln_eps_);
    h = ggml_mul(ctx, h, b.mlp_ln_w);
    h = ggml_add(ctx, h, b.mlp_ln_b);
    h = ggml_add(ctx, ggml_mul_mat(ctx, b.mlp0_w, h), b.mlp0_b);  // (5120, T)
    h = ggml_gelu(ctx, h);
    h = ggml_add(ctx, ggml_mul_mat(ctx, b.mlp2_w, h), b.mlp2_b);  // (1280, T)
    x = ggml_add(ctx, residual, h);
    char nm[32];
    if (il == 0)            { snprintf(nm,32,"dbg_blk0_out");  mark(x, nm); }
    else if (il == 1)       { snprintf(nm,32,"dbg_blk1_out");  mark(x, nm); }
    else if (il == 5)       { snprintf(nm,32,"dbg_blk5_out");  mark(x, nm); }
    else if (il == 11)      { snprintf(nm,32,"dbg_blk11_out"); mark(x, nm); }
  }

  // project_down (1280 → 8).
  ggml_tensor *y = ggml_add(ctx, ggml_mul_mat(ctx, pd_w_, x), pd_b_);
  ggml_set_name(y, "v3tok_proj_down");
  ggml_build_forward_expand(gf, y);
  return y;
}

// =====================================================================
// Tokenize — log-mel → encoder graph → FSQ encode
// =====================================================================

int CosyVoice3SpeechTokenizer::Tokenize(const float *pcm_16k, int n_samples,
                                        std::vector<int32_t> &tokens_out,
                                        ggml_backend_sched_t sched) {
  tokens_out.clear();
  if (!loaded_) {
    RS_LOG_ERR("Tokenizer: not loaded");
    return -1;
  }
  if (!pcm_16k || n_samples <= 0 || !sched) {
    RS_LOG_ERR("Tokenizer: bad inputs (n=%d)", n_samples);
    return -1;
  }

  // 1) log-mel features.
  static thread_local CosyVoice3MelExtractor mel_extr;
  std::vector<float> mel;
  int T_mel = mel_extr.Compute(pcm_16k, n_samples, mel);
  if (T_mel <= 0) {
    RS_LOG_ERR("Tokenizer: mel extraction failed");
    return -1;
  }
  // DEBUG: optionally override the in-house mel with a Python-computed
  // reference dump for parity verification.
  if (const char *p = std::getenv("RS_CV3_MEL_DUMP")) {
    FILE *f = std::fopen(p, "rb");
    if (f) {
      std::fseek(f, 0, SEEK_END);
      long sz = std::ftell(f);
      std::fseek(f, 0, SEEK_SET);
      const int n_floats = (int)(sz / sizeof(float));
      mel.assign((size_t)n_floats, 0.f);
      std::fread(mel.data(), sizeof(float), n_floats, f);
      std::fclose(f);
      T_mel = n_floats / 128;
      RS_LOG_INFO("Tokenizer: loaded Python mel dump (%d floats, T=%d)",
                  n_floats, T_mel);
    }
  }
  // Debug: dump first/last mel values for parity check vs ONNX-side mel.
  {
    float vmin = 1e9, vmax = -1e9, vsum = 0;
    for (float v : mel) {
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
      vsum += v;
    }
    RS_LOG_INFO("Tokenizer mel: T=%d shape=(128,%d) range=[%.3f,%.3f] mean=%.4f "
                "first5=[%.3f %.3f %.3f %.3f %.3f]",
                T_mel, T_mel, (double)vmin, (double)vmax,
                (double)(vsum / (mel.empty() ? 1 : mel.size())),
                (double)mel[0], (double)mel[1], (double)mel[2],
                (double)mel[3], (double)mel[4]);
  }
  // DEBUG: optionally dump the computed mel for offline comparison.
  if (const char *p = std::getenv("RS_CV3_DUMP_MEL128")) {
    if (FILE *f = std::fopen(p, "wb")) {
      std::fwrite(mel.data(), sizeof(float), mel.size(), f);
      std::fclose(f);
      RS_LOG_INFO("Tokenizer: dumped 128-mel (%zu floats, T=%d) -> %s",
                  mel.size(), T_mel, p);
    }
  }

  // 2) Build encoder graph + compute.
  constexpr int MAX_NODES = 16384;
  const size_t mem_size = MAX_NODES * (sizeof(ggml_tensor) + 256);
  ggml_init_params ip = { mem_size, nullptr, true };
  ggml_context *ctx = ggml_init(ip);
  if (!ctx) {
    RS_LOG_ERR("Tokenizer: ggml_init failed");
    return -1;
  }
  ggml_cgraph *gf = ggml_new_graph_custom(ctx, MAX_NODES, false);

  // The mel from CosyVoice3MelExtractor is laid out (n_mels=128, T_mel) with
  // T_mel-fast (row-major C-order: data[m * T_mel + t]). For the conv1d in
  // chatterbox-style input layout we want (T_mel, n_mels=128), so we swap
  // the ne values when creating the tensor.
  ggml_tensor *mel_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T_mel, n_mels_);
  ggml_set_input(mel_in);
  ggml_set_name(mel_in, "v3tok_mel_in");

  ggml_tensor *y = BuildEncoderGraph(ctx, gf, mel_in, T_mel);
  const int T_tok = (int)y->ne[1];

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("Tokenizer: alloc_graph failed");
    ggml_free(ctx);
    return -1;
  }

  // The mel from extractor is (n_mels, T_mel) row-major; we declared mel_in
  // as ne=(T_mel, n_mels) but the on-disk byte order matches because ggml
  // ne[0] is the fast-changing axis (= n_mels stride-1 in extractor, but
  // T_mel-stride-1 in our tensor).
  // To get the right layout we must transpose into a buffer with T_mel-fast
  // ordering before uploading.
  std::vector<float> mel_t_major((size_t)T_mel * n_mels_, 0.f);
  for (int m = 0; m < n_mels_; ++m) {
    for (int t = 0; t < T_mel; ++t) {
      // Source: extractor's (n_mels, T_mel) → data[m * T_mel + t]
      // Target: (T_mel, n_mels) with ne[0]=T_mel, so memory order is
      //         m-slow / t-fast → data[m * T_mel + t]
      // Wait — they're the SAME layout. Confirmed:
      //   extractor: out[m * T_mel + t]   (m slow, t fast)
      //   tensor:    ne[0]=T_mel, ne[1]=n_mels → memory data[ne1 * ne0 + ne0]
      //              = data[m * T_mel + t]    (m slow, t fast) ✓
      mel_t_major[(size_t)m * T_mel + t] = mel[(size_t)m * T_mel + t];
    }
  }
  ggml_backend_tensor_set(mel_in, mel_t_major.data(), 0,
                          mel_t_major.size() * sizeof(float));

  // Fill positions input for RoPE.
  if (use_rope_) {
    ggml_tensor *pos_t = ggml_graph_get_tensor(gf, "v3tok_positions");
    if (pos_t) {
      std::vector<int32_t> positions((size_t)pos_t->ne[0]);
      for (int i = 0; i < (int)positions.size(); ++i) positions[i] = i;
      ggml_backend_tensor_set(pos_t, positions.data(), 0,
                              positions.size() * sizeof(int32_t));
    }
  }

  if (ggml_backend_sched_graph_compute(sched, gf) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("Tokenizer: compute failed");
    ggml_backend_sched_reset(sched);
    ggml_free(ctx);
    return -1;
  }

  // 3) Pull back (proj_dim=8, T_tok) → FSQ encode on host.
  std::vector<float> proj((size_t)proj_dim_ * T_tok, 0.f);
  ggml_backend_tensor_get(y, proj.data(), 0, proj.size() * sizeof(float));

  // Optional debug: dump intermediate tensors + stats for parity check.
  if (std::getenv("RS_CV3_TOK_DEBUG")) {
    static const char *kDebugNames[] = {
        "dbg_post_conv1_gelu", "dbg_post_conv2_gelu", "dbg_transposed",
        "dbg_blk0_attn_ln", "dbg_blk0_Q", "dbg_blk0_K", "dbg_blk0_V",
        "dbg_blk0_fsmn_out", "dbg_blk0_out", "dbg_blk1_out",
        "dbg_blk5_out", "dbg_blk11_out",
    };
    for (const char *nm : kDebugNames) {
      ggml_tensor *t = ggml_graph_get_tensor(gf, nm);
      if (!t) continue;
      size_t n = ggml_nelements(t);
      std::vector<float> data((size_t)n, 0.f);
      ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));
      double sum = 0, sq = 0;
      for (auto v : data) { sum += v; sq += (double)v * v; }
      double mean = sum / n;
      double std_ = std::sqrt(sq / n - mean * mean);
      RS_LOG_INFO("dbg %-22s shape=[%lld,%lld,%lld,%lld] first3=[%.4f %.4f %.4f] mean=%.4f std=%.4f",
                  nm, (long long)t->ne[0], (long long)t->ne[1],
                  (long long)t->ne[2], (long long)t->ne[3],
                  (double)data[0],
                  (n > 1 ? (double)data[1] : 0.0),
                  (n > 2 ? (double)data[2] : 0.0),
                  mean, std_);
    }
  }
  // Reset before freeing ctx so the scheduler drops references to tensors
  // (mel_in / intermediates) that are about to disappear.
  ggml_backend_sched_reset(sched);
  ggml_free(ctx);

  // FSQ encode:  h = tanh(h) · gain → round-half-to-even → +1  (in {0,1,2})
  //              token = Σᵢ 3ⁱ · hᵢ   ∈ [0, 6561)
  constexpr int kPow[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
  tokens_out.assign((size_t)T_tok, 0);
  for (int t = 0; t < T_tok; ++t) {
    // proj layout after readback: ne=(8, T_tok) with ne[0]=8 fast →
    // memory data[t * 8 + i]
    const float *row = proj.data() + (size_t)t * proj_dim_;
    int code = 0;
    for (int i = 0; i < proj_dim_; ++i) {
      float h = std::tanh(row[i]) * fsq_gain_;
      int v = (int)std::nearbyint(h) + 1;
      if (v < 0) v = 0;
      if (v > 2) v = 2;
      code += v * kPow[i];
    }
    if (code < 0) code = 0;
    if (code >= codebook_) code = codebook_ - 1;
    tokens_out[t] = code;
  }
  return 0;
}
