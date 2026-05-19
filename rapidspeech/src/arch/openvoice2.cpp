#include "openvoice2.h"
#include "core/rs_context.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <numeric>
#include <random>

#define OV2_MAX_NODES 8192

// Duration predictor outputs are calibrated for a different frame resolution
// than the hop_length in this model.  Scaling factor to map raw predictions to
// natural mel-frame durations.

// =====================================================================
// Helper: Conv1D using ggml_conv_1d
// Input x is [in_channels, T], weight is [out_channels, in_channels, kw].
// ggml_conv_1d expects data in [OW, IC, N] layout (ne[1]==IC).
// =====================================================================
static struct ggml_tensor *conv1d_im2col(struct ggml_context *ctx,
                                          struct ggml_tensor *x,
                                          struct ggml_tensor *weight,
                                          struct ggml_tensor *bias,
                                          int kernel_size, int padding,
                                          int in_channels, int out_channels,
                                          int dilation = 1) {
  (void)kernel_size; (void)in_channels;  // kept for API compatibility
  struct ggml_tensor *x_t = ggml_cont(ctx, ggml_transpose(ctx, x));

  // CPU im2col requires F16 input; cast if needed
  struct ggml_tensor *x_in = x_t;
  struct ggml_tensor *w_in = weight;
  if (x_t->type == GGML_TYPE_F32 && weight->type == GGML_TYPE_F32) {
    x_in = ggml_cpy(ctx, x_t, ggml_new_tensor_2d(ctx, GGML_TYPE_F16, x_t->ne[0], x_t->ne[1]));
    w_in = ggml_cpy(ctx, weight, ggml_new_tensor_3d(ctx, GGML_TYPE_F16, weight->ne[0], weight->ne[1], weight->ne[2]));
  }

  struct ggml_tensor *out = ggml_conv_1d(ctx, w_in, x_in, 1, padding, dilation);

  // Cast output back to F32 if needed
  if (out->type != GGML_TYPE_F32) {
    out = ggml_cpy(ctx, out, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, out->ne[0], out->ne[1]));
  }

  out = ggml_cont(ctx, ggml_transpose(ctx, out));
  out = ggml_reshape_2d(ctx, out, out->ne[0], out->ne[1]);

  if (bias) {
    struct ggml_tensor *bias_2d = ggml_reshape_2d(ctx, bias, out_channels, 1);
    out = ggml_add(ctx, out, bias_2d);
  }

  return out;
}

// =====================================================================
// Helper: Weight Normalization
//
// Reconstructs the effective weight from weight_v and weight_g stored by
// torch.nn.utils.weight_norm.  The original weight is:
//     weight = g * v / ||v||
// where ||v|| is the L2 norm computed over all dims except the first
// (output-channel dim in PyTorch convention).  In GGUF the shapes are
// reversed, so weight_v is [ne0, ne1, ne2] and the norm is over ne0*ne1
// for each ne2 channel.
// =====================================================================
static struct ggml_tensor *apply_weight_norm(struct ggml_context *ctx,
                                              struct ggml_tensor *w_v,
                                              struct ggml_tensor *w_g) {
  int kw = (int)w_v->ne[0];  // kernel (or 1 for 1x1 conv)
  int md = (int)w_v->ne[1];  // middle dim
  int ch = (int)w_v->ne[2];  // per-channel count
  enum ggml_type orig_type = w_v->type;

  // Flatten norm dims: [kw * md, ch]
  int n = kw * md;
  struct ggml_tensor *w_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, w_v), n, ch);

  // Cast both to F32 for stable math (ggml_sum_rows requires F32)
  struct ggml_tensor *w_f32 = ggml_cast(ctx, w_2d, GGML_TYPE_F32);

  // ||v|| per channel: sqrt(sum(v^2, axis=0))
  struct ggml_tensor *sq    = ggml_sqr(ctx, w_f32);          // [n, ch]
  struct ggml_tensor *sum_sq = ggml_sum_rows(ctx, sq);      // [1, ch]
  struct ggml_tensor *norm  = ggml_sqrt(ctx, sum_sq);       // [1, ch]

  // Broadcast norm to match w_f32: [n, ch]
  struct ggml_tensor *norm_r = ggml_repeat(ctx, norm, w_f32);

  // v_norm = v / ||v||  →  [n, ch]
  struct ggml_tensor *v_norm = ggml_div(ctx, w_f32, norm_r);

  // Flatten weight_g from [1,1,ch] to [1,ch], cast to F32, broadcast
  struct ggml_tensor *g_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, w_g), 1, ch);
  struct ggml_tensor *g_f32 = ggml_cast(ctx, g_2d, GGML_TYPE_F32);
  struct ggml_tensor *g_r  = ggml_repeat(ctx, g_f32, v_norm);  // [n, ch]

  // weight = v_norm * g
  struct ggml_tensor *weight = ggml_mul(ctx, v_norm, g_r);    // [n, ch]

  // Cast back to original type
  weight = ggml_cast(ctx, weight, orig_type);

  // Reshape back to original 3-D shape
  weight = ggml_reshape_3d(ctx, weight, kw, md, ch);
  return weight;
}

// =====================================================================
// Helper: Layer Normalization
// =====================================================================
static struct ggml_tensor *layer_norm(struct ggml_context *ctx,
                                       struct ggml_tensor *x,
                                       struct ggml_tensor *w,
                                       struct ggml_tensor *b, float eps) {
  x = ggml_norm(ctx, x, eps);
  x = ggml_mul(ctx, x, w);
  if (b) x = ggml_add(ctx, x, b);
  return x;
}

// =====================================================================
// Pending-inputs registry for the no_alloc graph-building pattern.
//
// When building graphs with no_alloc=true (init_compute_ctx), input tensors
// have data=nullptr. We register the intended data here and upload it after
// ggml_backend_sched_alloc_graph via flush_pending_inputs().
// =====================================================================

struct PendingInput {
    struct ggml_tensor *tensor;
    std::vector<uint8_t> data;
};

static thread_local std::vector<PendingInput> g_pending_inputs;

static void register_pending_input(struct ggml_tensor *tensor,
                                    const void *data, size_t size) {
    PendingInput pi;
    pi.tensor = tensor;
    pi.data.resize(size);
    if (size > 0) std::memcpy(pi.data.data(), data, size);
    g_pending_inputs.push_back(std::move(pi));
}

static void flush_pending_inputs() {
    for (size_t i = 0; i < g_pending_inputs.size(); i++) {
        auto &pi = g_pending_inputs[i];
        ggml_backend_tensor_set(pi.tensor, pi.data.data(), 0, pi.data.size());
        if (pi.data.size() >= 4) {
            float first_val;
            ggml_backend_tensor_get(pi.tensor, &first_val, 0, sizeof(float));
        }
    }
    g_pending_inputs.clear();
}

// =====================================================================
// Helper: Multi-head Self-Attention with Relative Position (Shaw et al. 2018)
//
// emb_rel_k [head_dim, window_size]: Q-dependent relative position bias.
//   bias[i,j,h,b] = dot(Q[:,i,h,b], emb_rel_k[:, clip(i-j)])
//
// emb_rel_v [head_dim, window_size]: relative position value correction.
//   correction[i,d,h,b] = sum_j attn[i,j,h,b] * emb_rel_v[d, clip(i-j)]
//
// Context must be no_alloc=true (init_compute_ctx). Index tensors for
// get_rows are registered via register_pending_input and must be flushed
// by the caller after ggml_backend_sched_alloc_graph.
// =====================================================================
static struct ggml_tensor *multi_head_attention(
    struct ggml_context *ctx, struct ggml_tensor *x,
    struct ggml_tensor *q_w, struct ggml_tensor *q_b,
    struct ggml_tensor *k_w, struct ggml_tensor *k_b,
    struct ggml_tensor *v_w, struct ggml_tensor *v_b,
    struct ggml_tensor *o_w, struct ggml_tensor *o_b,
    struct ggml_tensor *emb_rel_k,
    struct ggml_tensor *emb_rel_v,
    int n_heads, int head_dim, int n_ctx) {

  int hidden = n_heads * head_dim;
  int B = x->ne[2] > 0 ? x->ne[2] : 1;


  // Squeeze kernel dim from 1×1 conv weights: [kw=1, in, out] → [in, out]
  // NOTE: weight tensors are from gguf_data context; ggml_cont copies them
  // into the compute ctx for safe reshaping
  if (q_w && q_w->ne[0] == 1 && q_w->ne[2] > 0)
    q_w = ggml_reshape_2d(ctx, ggml_cont(ctx, q_w), q_w->ne[1], q_w->ne[2]);
  if (k_w && k_w->ne[0] == 1 && k_w->ne[2] > 0)
    k_w = ggml_reshape_2d(ctx, ggml_cont(ctx, k_w), k_w->ne[1], k_w->ne[2]);
  if (v_w && v_w->ne[0] == 1 && v_w->ne[2] > 0)
    v_w = ggml_reshape_2d(ctx, ggml_cont(ctx, v_w), v_w->ne[1], v_w->ne[2]);
  if (o_w && o_w->ne[0] == 1 && o_w->ne[2] > 0)
    o_w = ggml_reshape_2d(ctx, ggml_cont(ctx, o_w), o_w->ne[1], o_w->ne[2]);


  // Linear projections
  struct ggml_tensor *Q = ggml_mul_mat(ctx, q_w, x);
  if (q_b) Q = ggml_add(ctx, Q, q_b);

  struct ggml_tensor *K = ggml_mul_mat(ctx, k_w, x);
  if (k_b) K = ggml_add(ctx, K, k_b);

  struct ggml_tensor *V = ggml_mul_mat(ctx, v_w, x);
  if (v_b) V = ggml_add(ctx, V, v_b);

  // Reshape to [head_dim, n_heads, n_ctx, B] → permute to [head_dim, n_ctx, n_heads, B]
  Q = ggml_permute(ctx,
      ggml_reshape_4d(ctx, Q, head_dim, n_heads, n_ctx, B), 0, 2, 1, 3);
  K = ggml_permute(ctx,
      ggml_reshape_4d(ctx, K, head_dim, n_heads, n_ctx, B), 0, 2, 1, 3);
  V = ggml_permute(ctx,
      ggml_reshape_4d(ctx, V, head_dim, n_heads, n_ctx, B), 0, 2, 1, 3);

  // Scaled dot-product attention
  float scale = 1.0f / sqrtf((float)head_dim);
  struct ggml_tensor *QK = ggml_mul_mat(ctx, Q, K);  // [n_ctx, n_ctx, n_heads, B]

  // --- Q-dependent relative position bias (emb_rel_k) ---
  if (emb_rel_k) {
    // Detect pre-computed bias matrix [n_ctx, n_ctx] vs weight tensor [head_dim, window_size]
    if ((int)emb_rel_k->ne[0] == n_ctx && (int)emb_rel_k->ne[1] == n_ctx) {
      // Pre-computed bias: broadcast [n_ctx, n_ctx, 1, 1] → [n_ctx, n_ctx, n_heads, B]
      struct ggml_tensor *bias_4d = ggml_reshape_4d(ctx, emb_rel_k, n_ctx, n_ctx, 1, 1);
      struct ggml_tensor *rel_bias = ggml_repeat(ctx, bias_4d, QK);
      QK = ggml_add(ctx, QK, rel_bias);
    } else {
      // Q-dependent relative position bias: bias = Q @ emb_rel_k, scattered to [n_ctx,n_ctx]
      int window_size = (int)emb_rel_k->ne[1];
      int max_rel_pos = (window_size - 1) / 2;

      // Compute Q_dot = Q^T @ emb_rel_k  →  [n_ctx*n_heads*B, window_size]
      // Q is permuted and not contiguous; make it contiguous before reshape
      struct ggml_tensor *Q_2d = ggml_reshape_2d(ctx, ggml_cont(ctx, Q),
          head_dim, n_ctx * n_heads * B);
      struct ggml_tensor *erk = ggml_reshape_2d(ctx,
          ggml_cont(ctx, emb_rel_k), head_dim, window_size);
      struct ggml_tensor *Q_dot = ggml_mul_mat(ctx, Q_2d, erk);

      // Reshape to [window_size, n_ctx, n_heads, B] then permute to [n_ctx, window_size, n_heads, B]
      Q_dot = ggml_reshape_4d(ctx, Q_dot, window_size, n_ctx, n_heads, B);
      Q_dot = ggml_cont(ctx, ggml_permute(ctx, Q_dot, 1, 0, 2, 3));

      // Flatten, then reshape to [1, N] so each scalar is a 1-element row
      int64_t qdot_total = (int64_t)n_ctx * window_size * n_heads * B;
      struct ggml_tensor *Q_dot_rows = ggml_reshape_2d(ctx,
          ggml_reshape_1d(ctx, Q_dot, qdot_total), 1, qdot_total);

      // Precompute flat indices for gathering bias[i,j,h,b] = Q_dot[i, r, h, b]
      // where r = clip(i-j) + max_rel_pos
      int total_bias = n_ctx * n_ctx * n_heads * B;
      std::vector<int32_t> flat_indices(total_bias);
      int idx_pos = 0;
      for (int i = 0; i < n_ctx; i++) {
        for (int j = 0; j < n_ctx; j++) {
          int r = i - j;
          if (r < -max_rel_pos) r = -max_rel_pos;
          if (r > max_rel_pos) r = max_rel_pos;
          r += max_rel_pos;
          for (int h = 0; h < n_heads; h++) {
            for (int b = 0; b < B; b++) {
              flat_indices[idx_pos++] = (int32_t)(i +
                  n_ctx * ((int64_t)r + window_size * ((int64_t)h + n_heads * (int64_t)b)));
            }
          }
        }
      }

      struct ggml_tensor *idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_bias);
      ggml_set_name(idx, "rel_k_idx");
      ggml_set_input(idx);
      register_pending_input(idx, flat_indices.data(),
                             total_bias * sizeof(int32_t));

      // get_rows on [1, qdot_total] with idx gives [1, total_bias]
      struct ggml_tensor *bias_1d = ggml_get_rows(ctx, Q_dot_rows, idx);
      struct ggml_tensor *rel_bias = ggml_reshape_4d(ctx,
          ggml_reshape_1d(ctx, bias_1d, total_bias),
          n_ctx, n_ctx, n_heads, B);
      QK = ggml_add(ctx, QK, rel_bias);
    }
  }

  // Softmax over KEY dimension: transpose QK from [Tq, Tk, nh, B] to [Tk, Tq, nh, B]
  // so ggml_soft_max_ext normalizes over ne0 (key dim).
  // Keep [Tk, Tq] layout for correct contraction with V^T in next step.
  struct ggml_tensor *QK_t = ggml_cont(ctx, ggml_permute(ctx, QK, 1, 0, 2, 3));
  struct ggml_tensor *QK_soft = ggml_soft_max_ext(ctx, QK_t, nullptr, scale, 0.0f);

  // Attention output: attn @ V
  struct ggml_tensor *QKV = ggml_mul_mat(
      ctx, ggml_cont(ctx, ggml_transpose(ctx, V)), QK_soft);

  // --- Relative position VALUE correction (emb_rel_v) ---
  if (emb_rel_v) {
    int window_size = (int)emb_rel_v->ne[1];
    int max_rel_pos = (window_size - 1) / 2;

    // Build attn_by_rel[r, i, h, b] = attn[i, i-r, h, b] for each (i, r, h, b).
    // QK_soft: [n_ctx, n_ctx, n_heads, B], contiguous
    // Flat index of element [i, j, h, b]: i + n_ctx*(j + n_ctx*(h + n_heads*b))
    int64_t attn_total = (int64_t)n_ctx * n_ctx * n_heads * B;
    struct ggml_tensor *attn_rows = ggml_reshape_2d(ctx,
        ggml_reshape_1d(ctx, ggml_cont(ctx, QK_soft), attn_total),
        1, attn_total);

    int total_pairs = n_ctx * window_size * n_heads * B;
    std::vector<int32_t> rv_idx_data(total_pairs);
    std::vector<float>   rv_mask_data(total_pairs, 0.0f);
    int pi = 0;
    for (int i = 0; i < n_ctx; i++) {
      for (int r_idx = 0; r_idx < window_size; r_idx++) {
        int r = r_idx - max_rel_pos;
        int j = i - r;
        for (int h = 0; h < n_heads; h++) {
          for (int b = 0; b < B; b++) {
            if (j >= 0 && j < n_ctx) {
              rv_idx_data[pi] = (int32_t)(i +
                  n_ctx * ((int64_t)j + n_ctx * ((int64_t)h + n_heads * (int64_t)b)));
              rv_mask_data[pi] = 1.0f;
            } else {
              rv_idx_data[pi] = 0;  // out of bounds, will be masked
              rv_mask_data[pi] = 0.0f;
            }
            pi++;
          }
        }
      }
    }

    struct ggml_tensor *idx_v = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_pairs);
    ggml_set_name(idx_v, "rel_v_idx");
    ggml_set_input(idx_v);
    register_pending_input(idx_v, rv_idx_data.data(),
                           total_pairs * sizeof(int32_t));

    struct ggml_tensor *mask_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, total_pairs);
    ggml_set_name(mask_v, "rel_v_mask");
    ggml_set_input(mask_v);
    register_pending_input(mask_v, rv_mask_data.data(),
                           total_pairs * sizeof(float));

    // get_rows on [1, attn_total] → [1, total_pairs] → apply mask → [window_size, n_ctx, n_heads, B]
    struct ggml_tensor *attn_gathered = ggml_get_rows(ctx, attn_rows, idx_v);
    struct ggml_tensor *attn_flat = ggml_reshape_1d(ctx, attn_gathered, total_pairs);
    attn_flat = ggml_mul(ctx, attn_flat,
        ggml_reshape_1d(ctx, mask_v, total_pairs));
    struct ggml_tensor *attn_by_rel = ggml_reshape_4d(ctx, attn_flat,
        n_ctx, window_size, n_heads, B);
    // Permute to [window_size, n_ctx, n_heads, B] for mul_mat
    attn_by_rel = ggml_permute(ctx, attn_by_rel, 1, 0, 2, 3);

    // correction = erv @ attn_by_rel: [win, head_dim] @ [win, n_ctx, nh, B]
    // → [head_dim, n_ctx, n_heads, B]
    // Copy emb_rel_v into compute context via ggml_cont
    struct ggml_tensor *erv = ggml_reshape_2d(ctx,
        ggml_cont(ctx, emb_rel_v), head_dim, window_size);
    erv = ggml_cont(ctx, ggml_transpose(ctx, erv));  // → [win, head_dim]
    struct ggml_tensor *rel_v_corr = ggml_mul_mat(ctx, erv, attn_by_rel);
    // result is already [head_dim, n_ctx, nh, B] — no permute needed
    QKV = ggml_add(ctx, QKV, rel_v_corr);
  }

  // Merge heads: permute & reshape back to [hidden, n_ctx, B]
  struct ggml_tensor *attn = ggml_reshape_3d(
      ctx,
      ggml_cont(ctx, ggml_permute(ctx, QKV, 0, 2, 1, 3)),
      hidden, n_ctx, B);

  // Output projection
  attn = ggml_mul_mat(ctx, o_w, attn);
  if (o_b) attn = ggml_add(ctx, attn, o_b);

  return attn;
}

// =====================================================================
// OpenVoice2Model implementation
// =====================================================================

OpenVoice2Model::OpenVoice2Model() {}
OpenVoice2Model::~OpenVoice2Model() {}

bool OpenVoice2Model::MapTensors(std::map<std::string, struct ggml_tensor*>& all) {
  for (auto& [name, tensor] : all) {
    if (name.find("text_encoder.") == 0) {
      weights_.text_encoder[name] = tensor;
    } else if (name.find("duration_predictor.") == 0) {
      weights_.duration_predictor[name] = tensor;
    } else if (name.find("flow_decoder.") == 0) {
      weights_.flow_decoder[name] = tensor;
    } else if (name.find("vocoder.") == 0) {
      weights_.vocoder[name] = tensor;
    } else if (name.find("posterior_encoder.") == 0) {
      weights_.posterior_encoder[name] = tensor;
    } else if (name.find("emb_") == 0) {
      weights_.embeddings[name] = tensor;
    } else {
      // Store in text_encoder as fallback (some weights may not have prefix)
      weights_.text_encoder[name] = tensor;
    }
  }
  RS_LOG_INFO("OpenVoice2: mapped %zu text_enc, %zu dur_pred, %zu flow, %zu vocoder tensors",
              weights_.text_encoder.size(), weights_.duration_predictor.size(),
              weights_.flow_decoder.size(), weights_.vocoder.size());
  return true;
}

bool OpenVoice2Model::Load(const std::unique_ptr<rs_context_t>& ctx,
                            ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("Invalid context for OpenVoice2 Load");
    return false;
  }

  gguf_context* ctx_gguf = ctx->ctx_gguf;
  ggml_context* gguf_data = ctx->gguf_data;

  // Load hyperparameters from GGUF KV (metadata written by conversion script)
  auto read_i32 = [&](const char* kk, int32_t& dst) {
    int64_t key = gguf_find_key(ctx_gguf, kk);
    if (key != -1) dst = gguf_get_val_i32(ctx_gguf, key);
  };
  read_i32("openvoice2.hidden_channels", hparams_.hidden_channels);
  read_i32("openvoice2.inter_channels",  hparams_.inter_channels);
  read_i32("openvoice2.filter_channels", hparams_.filter_channels);
  read_i32("openvoice2.n_heads",         hparams_.n_heads);
  read_i32("openvoice2.n_layers",        hparams_.n_layers);
  // n_flow_layers is auto-detected from weight names (NOT from GGUF metadata,
  // which stores n_layers_trans_flow — transformer layers per coupling block).
  read_i32("openvoice2.n_layers_trans_flow", hparams_.n_flow_layers);  // ignored if not present
  read_i32("openvoice2.sample_rate",     hparams_.sample_rate);
  read_i32("openvoice2.hop_length",      hparams_.hop_length);
  read_i32("openvoice2.n_fft",           hparams_.n_fft);
  read_i32("openvoice2.n_mels",          hparams_.n_mels);
  read_i32("openvoice2.num_tones",       hparams_.num_tones);
  read_i32("openvoice2.num_languages",   hparams_.num_languages);
  read_i32("openvoice2.vocab_size",      hparams_.vocab_size);

  meta_.arch_name = "openvoice2";
  meta_.audio_sample_rate = hparams_.sample_rate;
  meta_.n_mels = hparams_.n_mels;
  meta_.vocab_size = hparams_.vocab_size;

  RS_LOG_INFO("OpenVoice2: hidden=%d, sr=%d, hop=%d",
              hparams_.hidden_channels, hparams_.sample_rate, hparams_.hop_length);

  // Init text frontend — try GGUF symbol table(s) first, fall back to built-in
  {
    auto try_parse_symbols = [&](const std::string& raw) -> bool {
      if (raw.empty()) return false;
      std::vector<std::string> symbols;
      // Try JSON array format: ["_", "AA", ...]
      if (raw[0] == '[') {
        size_t pos = 1;
        while (pos < raw.size()) {
          if (raw[pos] == ']') break;
          if (raw[pos] == ',' || raw[pos] == ' ' || raw[pos] == '\n') { pos++; continue; }
          if (raw[pos] == '"') {
            pos++;
            std::string sym;
            while (pos < raw.size() && raw[pos] != '"') {
              if (raw[pos] == '\\' && pos + 1 < raw.size()) pos++;
              sym += raw[pos++];
            }
            if (pos < raw.size()) pos++;
            symbols.push_back(sym);
          } else { pos++; }
        }
      } else {
        // Comma-separated format: AA,E,EE,...
        size_t start = 0;
        while (start < raw.size()) {
          size_t end = raw.find(',', start);
          if (end == std::string::npos) end = raw.size();
          std::string sym = raw.substr(start, end - start);
          if (!sym.empty()) symbols.push_back(sym);
          start = end + 1;
        }
      }
      if (!symbols.empty()) {
        text_frontend_.InitFromSymbols(symbols);
        RS_LOG_INFO("OpenVoice2: loaded %zu symbols from GGUF metadata", symbols.size());
        return true;
      }
      return false;
    };

    bool symbols_loaded = false;
    // Try standard tokenizer key first (JSON array)
    const int tok_sym_key = gguf_find_key(ctx_gguf, "tokenizer.ggml.symbols");
    if (tok_sym_key != -1) {
      symbols_loaded = try_parse_symbols(gguf_get_val_str(ctx_gguf, tok_sym_key));
    }
    // Fall back to architecture-specific key (comma-separated)
    if (!symbols_loaded) {
      const int arch_sym_key = gguf_find_key(ctx_gguf, "openvoice2.symbols");
      if (arch_sym_key != -1) {
        symbols_loaded = try_parse_symbols(gguf_get_val_str(ctx_gguf, arch_sym_key));
      }
    }
    if (!symbols_loaded) {
      text_frontend_.Init(nullptr);
      RS_LOG_WARN("OpenVoice2: no symbol table in GGUF, using built-in vocab (IDs may mismatch!)");
    }
  }

  // Map all tensors
  std::map<std::string, struct ggml_tensor*> tensors;
  const int n_tensors = gguf_get_n_tensors(ctx_gguf);
  for (int i = 0; i < n_tensors; ++i) {
    const char* name = gguf_get_tensor_name(ctx_gguf, i);
    struct ggml_tensor* t = ggml_get_tensor(gguf_data, name);
    if (t) tensors[name] = t;
  }

  if (!MapTensors(tensors)) return false;

  // Auto-detect n_mels from vocoder conv_pre input channels (fallback only)
  if (hparams_.n_mels <= 0 && weights_.vocoder.count("vocoder.conv_pre.weight")) {
    auto* pre_w = weights_.vocoder["vocoder.conv_pre.weight"];
    hparams_.n_mels = static_cast<int32_t>(pre_w->ne[1]);  // in_channels
    RS_LOG_INFO("OpenVoice2: auto-detected n_mels=%d", hparams_.n_mels);
  }

  // Auto-detect n_flow_layers from flow_decoder weights
  // Chinese MeloTTS: flows at even indices have transformer encoders,
  // odd indices are flip-only. Total flows = max_flow_idx + 2 (for final flip).
  {
    int max_flow_idx = -1;
    for (auto& [name, t] : weights_.flow_decoder) {
      if (name.find("flow_decoder.flows.") == 0) {
        // name like "flow_decoder.flows.N.xxx"
        size_t pos = strlen("flow_decoder.flows.");
        int idx = 0;
        const char* p = name.c_str() + pos;
        while (*p >= '0' && *p <= '9') { idx = idx * 10 + (*p - '0'); p++; }
        if (idx > max_flow_idx) max_flow_idx = idx;
      }
    }
    if (max_flow_idx >= 0) {
      hparams_.n_flow_layers = max_flow_idx + 2;  // +1 for 0-index, +1 for final flip
      RS_LOG_INFO("OpenVoice2: auto-detected n_flow_layers=%d (max_idx=%d)",
                  hparams_.n_flow_layers, max_flow_idx);
    }
  }

  return true;
}

bool OpenVoice2Model::LoadConverter(const char* converter_path,
                                     ggml_backend_t backend) {
  RS_LOG_INFO("OpenVoice2: converter loading not yet implemented: %s", converter_path);
  converter_weights_.loaded = false;
  return true;  // Non-fatal: base TTS works without converter
}

std::shared_ptr<RSState> OpenVoice2Model::CreateState() {
  return std::make_shared<OpenVoice2State>();
}

// =====================================================================
// TTS-specific methods
// =====================================================================

bool OpenVoice2Model::PushText(RSState& state, const char* text,
                                const char* language, const char* instruct) {
  auto& s = static_cast<OpenVoice2State&>(state);
  s.language = language ? language : "zh";

  // Set language_id based on language string
  if (s.language == "zh" || s.language == "ZH") s.language_id = 0;
  else if (s.language == "en" || s.language == "EN") s.language_id = 1;
  else if (s.language == "ja" || s.language == "JA") s.language_id = 2;
  else s.language_id = 3;  // "other"

  s.tone_ids.clear();
  s.phoneme_ids = text_frontend_.TextToPhonemeIds(text, s.language, &s.tone_ids);

  if (s.phoneme_ids.empty()) {
    RS_LOG_ERR("OpenVoice2: text frontend produced no phonemes");
    return false;
  }

  // Debug: print phoneme IDs
  std::string id_str;
  for (size_t i = 0; i < s.phoneme_ids.size(); i++) {
    if (i > 0) id_str += ", ";
    id_str += std::to_string(s.phoneme_ids[i]);
  }
  RS_LOG_INFO("OpenVoice2: text -> %zu phoneme IDs [%s], %zu tone IDs",
              s.phoneme_ids.size(), id_str.c_str(), s.tone_ids.size());
  return true;
}

bool OpenVoice2Model::PushReferenceAudio(RSState& state, const float* samples,
                                          int n_samples, int sample_rate,
                                          ggml_backend_sched_t sched) {
  auto& s = static_cast<OpenVoice2State&>(state);

  if (!converter_weights_.loaded) {
    RS_LOG_WARN("OpenVoice2: tone converter not loaded, ignoring reference audio");
    return true;  // Non-fatal
  }

  // TODO: compute mel spectrogram from reference audio
  // TODO: run tone color encoder to extract style embedding
  s.has_tone_embedding = false;
  return true;
}

// =====================================================================
// Encode: TextEncoder + DurationPredictor + FlowDecoder
// =====================================================================

bool OpenVoice2Model::Encode(const std::vector<float>& input_frames,
                              RSState& state, ggml_backend_sched_t sched) {
  auto& s = static_cast<OpenVoice2State&>(state);
  (void)input_frames;  // TTS doesn't use audio input for encoding

  if (s.phoneme_ids.empty()) {
    RS_LOG_ERR("OpenVoice2: no text pushed, call PushText first");
    return false;
  }

  // Step 1: Text Encoder
  if (!RunTextEncoder(s, sched)) return false;

  // Step 2: Duration Predictor
  if (!RunDurationPredictor(s, sched)) return false;

  // Step 2.5: Expand m_p / logs_p by durations, sample z_p
  {
    int C = hparams_.hidden_channels;
    int T_txt = s.encoder_T;
    int T_mel = s.total_mel_frames;

    s.z_p_expanded.resize(C * T_mel);
    std::vector<float> m_p_exp(C * T_mel, 0.0f);
    std::vector<float> lp_exp(C * T_mel, 0.0f);

    int mel_pos = 0;
    for (int t = 0; t < T_txt && mel_pos < T_mel; t++) {
      int dur = s.durations[t];
      for (int d = 0; d < dur && mel_pos < T_mel; d++) {
        for (int c = 0; c < C; c++) {
          m_p_exp[c + mel_pos * C]  = s.m_p[c + t * C];
          lp_exp[c + mel_pos * C]   = s.logs_p[c + t * C];
        }
        mel_pos++;
      }
    }

    // z_p = m_p + randn * exp(logs_p) * noise_scale
    // Clip logs_p to prevent exp() overflow (max float32 ≈ 3.4e38, exp(88) ≈ 1.6e38)
    constexpr float noise_scale = 0.667f;
    constexpr float lp_max = 80.0f;  // exp(80) ≈ 5.5e34, safe in float32
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < C * T_mel; i++) {
      float noise = dist(rng);
      float lp = lp_exp[i];
      if (lp > lp_max) lp = lp_max;
      s.z_p_expanded[i] = m_p_exp[i] + noise * std::exp(lp) * noise_scale;
    }

    float zp_min = 1e9, zp_max = -1e9;
    for (int i = 0; i < C * T_mel; i++) {
      if (s.z_p_expanded[i] < zp_min) zp_min = s.z_p_expanded[i];
      if (s.z_p_expanded[i] > zp_max) zp_max = s.z_p_expanded[i];
    }
    RS_LOG_INFO("OpenVoice2: z_p sampled [%.4f..%.4f] (%d values)",
                zp_min, zp_max, C * T_mel);
  }

  // Step 3: Flow Decoder (generates full mel spectrogram)
  if (!RunFlowDecoder(s, sched)) return false;

  // Reset streaming cursor
  s.mel_chunk_cursor = 0;
  s.audio_output.clear();
  s.audio_read_cursor = 0;

  return true;
}

// =====================================================================
// Decode: Vocoder on next mel chunk (streaming)
// =====================================================================

bool OpenVoice2Model::Decode(RSState& state, ggml_backend_sched_t sched) {
  auto& s = static_cast<OpenVoice2State&>(state);

  if (s.mel_spectrogram.empty() || s.mel_chunk_cursor >= s.total_mel_frames) {
    return false;  // No more chunks
  }

  int chunk_size = hparams_.chunk_mel_frames;
  if (chunk_size <= 0) chunk_size = s.total_mel_frames;  // Non-streaming

  int mel_start = s.mel_chunk_cursor;
  int mel_len = std::min(chunk_size, s.total_mel_frames - mel_start);

  if (!RunVocoder(s, sched, mel_start, mel_len)) return false;

  s.mel_chunk_cursor += mel_len;
  return true;
}

int OpenVoice2Model::GetAudioOutput(RSState& state, float** out_data) {
  auto& s = static_cast<OpenVoice2State&>(state);
  if (s.audio_read_cursor >= static_cast<int>(s.audio_output.size())) {
    *out_data = nullptr;
    return 0;
  }
  *out_data = s.audio_output.data() + s.audio_read_cursor;
  int n = static_cast<int>(s.audio_output.size()) - s.audio_read_cursor;
  s.audio_read_cursor = static_cast<int>(s.audio_output.size());
  return n;
}

// =====================================================================
// Sub-graph: Text Encoder (Transformer with relative-position self-attention)
//
// Architecture (VITS/MeloTTS text encoder):
//   1. Phoneme Embedding lookup
//   2. Scale + add positional encoding
//   3. N × Transformer block:
//      a. LayerNorm → Multi-head Self-Attention → Residual
//      b. LayerNorm → Conv1D+GELU+Conv1D (FFN) → Residual
//   4. Final LayerNorm
// =====================================================================

bool OpenVoice2Model::RunTextEncoder(OpenVoice2State& state,
                                      ggml_backend_sched_t sched) {
  auto& w = weights_.text_encoder;
  if (w.empty()) {
    RS_LOG_ERR("OpenVoice2: no text_encoder weights loaded");
    return false;
  }

  int T = static_cast<int>(state.phoneme_ids.size());
  int C = hparams_.hidden_channels;
  int n_heads = hparams_.n_heads;
  int head_dim = C / n_heads;
  int n_layers = hparams_.n_layers;

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for TextEncoder");
    return false;
  }

  g_pending_inputs.clear();

  // Input: phoneme IDs
  struct ggml_tensor *phoneme_ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
  ggml_set_name(phoneme_ids, "phoneme_ids");
  ggml_set_input(phoneme_ids);

  struct ggml_tensor *cur = nullptr;

  // --- Embedding Lookup ---
  // Try text_encoder.emb.weight first, then emb_g.weight
  struct ggml_tensor *emb_table = nullptr;
  for (auto& [name, t] : w) {
    if (name.find("emb.") != std::string::npos ||
        name.find("emb_") != std::string::npos) {
      emb_table = t;
      break;
    }
  }
  if (!emb_table) {
    // Fallback: look in embeddings map
    for (auto& [name, t] : weights_.embeddings) {
      emb_table = t;
      break;
    }
  }

  // Embedding table: [hidden, vocab] (ggml) or [vocab, hidden] (PyTorch)
  if (emb_table && (emb_table->ne[0] >= C || emb_table->ne[1] >= C)) {
    // DEBUG: read raw emb_table values to verify data integrity
    {
      int n0 = (int)emb_table->ne[0], n1 = (int)emb_table->ne[1];
      int n_read = std::min(10, n0 * n1);
      std::vector<float> raw_emb(n_read);
      ggml_backend_tensor_get(emb_table, raw_emb.data(), 0, n_read * sizeof(float));
      RS_LOG_INFO("OpenVoice2: emb_table [%d,%d] name=%s first10=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                  n0, n1, emb_table->name ? emb_table->name : "?",
                  raw_emb[0], raw_emb[1], raw_emb[2], raw_emb[3], raw_emb[4],
                  raw_emb[5], raw_emb[6], raw_emb[7], raw_emb[8], raw_emb[9]);
    }
    // PyTorch stores embeddings as [vocab, hidden] — transpose for ggml
    struct ggml_tensor *emb = emb_table;
    if (emb_table->ne[0] < C) {
      emb = ggml_transpose(ctx0, emb_table);
    }
    cur = ggml_get_rows(ctx0, emb, phoneme_ids);
    ggml_set_name(cur, "phoneme_emb");
  } else {
    // No embedding table available — create one-hot-like input directly
    // Use a zero tensor with identity-like values at phoneme positions
    cur = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
    ggml_set_input(cur);
    RS_LOG_WARN("OpenVoice2: no embedding table found, using raw input");
  }

  // --- Tone Embedding ---
  // tone_emb.weight: [hidden_channels, n_tones] where n_tones=11 (tone 0-5 + extras)
  struct ggml_tensor *tone_emb_weight = nullptr;
  auto tone_it = w.find("text_encoder.tone_emb.weight");
  if (tone_it != w.end()) tone_emb_weight = tone_it->second;

  if (tone_emb_weight && !state.tone_ids.empty()) {
    int Ttone = (int)state.tone_ids.size();
    struct ggml_tensor *tone_ids_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, Ttone);
    ggml_set_name(tone_ids_tensor, "tone_ids");
    ggml_set_input(tone_ids_tensor);

    register_pending_input(tone_ids_tensor, state.tone_ids.data(),
                           Ttone * sizeof(int32_t));

    // Ensure embedding table has ne[0]=C for proper get_rows output
    struct ggml_tensor *tone_tbl = tone_emb_weight;
    if (tone_tbl->ne[0] < C) tone_tbl = ggml_transpose(ctx0, tone_tbl);
    struct ggml_tensor *tone_emb = ggml_get_rows(ctx0, tone_tbl, tone_ids_tensor);
    ggml_set_name(tone_emb, "tone_emb");
    cur = ggml_add(ctx0, cur, tone_emb);
  }

  // --- Language Embedding ---
  // language_emb.weight: [hidden_channels, n_langs] where n_langs=4
  struct ggml_tensor *lang_emb_weight = nullptr;
  auto lang_it = w.find("text_encoder.language_emb.weight");
  if (lang_it != w.end()) lang_emb_weight = lang_it->second;

  if (lang_emb_weight) {
    // Repeat lang_id T times so get_rows produces [C, T] directly
    std::vector<int32_t> lang_ids_all(T, state.language_id);
    struct ggml_tensor *lang_id_tensor = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
    ggml_set_name(lang_id_tensor, "lang_ids");
    ggml_set_input(lang_id_tensor);

    register_pending_input(lang_id_tensor, lang_ids_all.data(),
                           T * sizeof(int32_t));

    struct ggml_tensor *lang_tbl = lang_emb_weight;
    if (lang_tbl->ne[0] < C) lang_tbl = ggml_transpose(ctx0, lang_tbl);
    struct ggml_tensor *lang_emb = ggml_get_rows(ctx0, lang_tbl, lang_id_tensor);
    ggml_set_name(lang_emb, "lang_emb");
    cur = ggml_add(ctx0, cur, lang_emb);
  }

  // --- Scale ---
  // MeloTTS embedding formula: (emb + tone_emb + lang_emb) * sqrt(hidden_channels)
  // No explicit positional encoding — position is handled by relative position bias.
  cur = ggml_scale(ctx0, cur, sqrtf((float)C));

  // Save reference to embedding output for debug readback
  struct ggml_tensor *emb_out_tensor = cur;

  // --- DEBUG: Run only N_DEBUG_LAYERS transformer layers ---
  #if 0
  #define DEBUG_LAYERS 6  // -2=test layer_norm only, -1=embed step-by-step, 0=embed only, 1..6=layers
  {
    if (DEBUG_LAYERS == -2) {
      // Test layer_norm only: verify ggml_norm produces correct output
      struct ggml_tensor *emb = emb_table;
      if (emb->ne[0] < C) emb = ggml_transpose(ctx0, emb);
      struct ggml_tensor *cur_test = ggml_get_rows(ctx0, emb, phoneme_ids);

      int n_in = 2;  // phoneme_ids + either tone_ids or PE
      struct ggml_tensor *tone_ids_t = nullptr, *lang_ids_t = nullptr, *pos_enc_t = nullptr;

      struct ggml_tensor *tone_tbl = nullptr, *lang_tbl = nullptr;
      auto it_tone = w.find("text_encoder.tone_emb.weight");
      if (it_tone != w.end()) tone_tbl = it_tone->second;
      if (tone_tbl) {
        if (tone_tbl->ne[0] < C) tone_tbl = ggml_transpose(ctx0, tone_tbl);
        tone_ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_input(tone_ids_t);
        ggml_set_name(tone_ids_t, "tone_in");
        struct ggml_tensor *te = ggml_get_rows(ctx0, tone_tbl, tone_ids_t);
        cur_test = ggml_add(ctx0, cur_test, te);
        n_in++;
      }

      auto it_lang = w.find("text_encoder.language_emb.weight");
      if (it_lang != w.end()) lang_tbl = it_lang->second;
      if (lang_tbl) {
        if (lang_tbl->ne[0] < C) lang_tbl = ggml_transpose(ctx0, lang_tbl);
        lang_ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_input(lang_ids_t);
        ggml_set_name(lang_ids_t, "lang_in");
        struct ggml_tensor *le = ggml_get_rows(ctx0, lang_tbl, lang_ids_t);
        cur_test = ggml_add(ctx0, cur_test, le);
        n_in++;
      }

      cur_test = ggml_scale(ctx0, cur_test, sqrtf((float)C));
      pos_enc_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
      ggml_set_input(pos_enc_t);
      ggml_set_name(pos_enc_t, "pe_in");
      n_in++;
      cur_test = ggml_add(ctx0, cur_test, pos_enc_t);
      ggml_set_name(cur_test, "emb_in");
      ggml_set_output(cur_test);

      // layer_norm
      struct ggml_tensor *n1w = nullptr, *n1b = nullptr;
      for (auto& [name, t] : w) {
        if (name.find("norm_layers_1.0.gamma") != std::string::npos) n1w = t;
        if (name.find("norm_layers_1.0.beta") != std::string::npos)  n1b = t;
      }
      if (!n1w) { RS_LOG_INFO("OpenVoice2: LN test — no norm1 weights found"); ggml_free(ctx0); return true; }
      RS_LOG_INFO("OpenVoice2: LN test — n1w ne0=%lld ne1=%lld", (long long)n1w->ne[0], (long long)n1w->ne[1]);
      struct ggml_tensor *normed = layer_norm(ctx0, cur_test, n1w, n1b, 1e-5f);
      ggml_set_output(normed);
      ggml_build_forward_expand(gf, normed);

      ggml_backend_sched_reset(sched);
      if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_free(ctx0); return false; }

      // Upload inputs
      ggml_backend_tensor_set(phoneme_ids, state.phoneme_ids.data(), 0, state.phoneme_ids.size()*sizeof(int32_t));
      if (tone_ids_t) {
        std::vector<int32_t> td = state.tone_ids;
        if (td.empty()) td.resize(T, 0);
        ggml_backend_tensor_set(tone_ids_t, td.data(), 0, td.size()*sizeof(int32_t));
      }
      if (lang_ids_t) {
        std::vector<int32_t> ld(T, state.language_id);
        ggml_backend_tensor_set(lang_ids_t, ld.data(), 0, ld.size()*sizeof(int32_t));
      }
      if (pos_enc_t) {
        std::vector<float> pd(C*T);
        for (int p = 0; p < T; p++)
          for (int i = 0; i < C/2; i++) {
            float freq = powf(10000.0f, -2.0f*(float)i/(float)C);
            pd[p*C + 2*i] = sinf((float)p * freq);
            pd[p*C + 2*i+1] = cosf((float)p * freq);
          }
        ggml_backend_tensor_set(pos_enc_t, pd.data(), 0, pd.size()*sizeof(float));
      }

      ggml_backend_sched_graph_compute(sched, gf);

      auto read_log = [](struct ggml_tensor *t, const char *lbl) {
        int64_t n = t->ne[0] * t->ne[1];
        std::vector<float> d(n);
        ggml_backend_tensor_get(t, d.data(), 0, d.size()*sizeof(float));
        float mn=1e9,mx=-1e9; double s=0;
        for(auto v:d){if(v<mn)mn=v;if(v>mx)mx=v;s+=v;}
        RS_LOG_INFO("OpenVoice2: %s [%lld,%lld] -> [%.4f..%.4f] mean=%.4f first5=%.4f,%.4f,%.4f,%.4f,%.4f",
            lbl,(long long)t->ne[0],(long long)t->ne[1],mn,mx,(float)(s/d.size()),d[0],d[1],d[2],d[3],d[4]);
      };
      read_log(cur_test, "LN_INPUT");
      read_log(normed, "LN_OUTPUT");

      ggml_free(ctx0);
      return true;
    }

    if (DEBUG_LAYERS < 0) {
      // Build single graph with 5 outputs tracing embedding pipeline:
      //   s1: get_rows(emb, phoneme_ids)
      //   s2: s1 + tone_emb
      //   s3: s2 + lang_emb
      //   s4: s3 * sqrt(C)
      //   s5: s4 + pos_enc
      struct ggml_tensor *emb = emb_table;
      if (emb->ne[0] < C) emb = ggml_transpose(ctx0, emb);
      struct ggml_tensor *step_emb = ggml_get_rows(ctx0, emb, phoneme_ids);
      ggml_set_name(step_emb, "s1_emb");
      ggml_set_output(step_emb);

      // Declare input tensor pointers outside blocks so we can reference them later
      struct ggml_tensor *tone_ids_t = nullptr;
      struct ggml_tensor *lang_id_t  = nullptr;
      struct ggml_tensor *pos_enc_t  = nullptr;

      struct ggml_tensor *step_tone = step_emb;
      struct ggml_tensor *tone_tbl = nullptr;
      auto it_tone = w.find("text_encoder.tone_emb.weight");
      if (it_tone != w.end()) tone_tbl = it_tone->second;
      if (tone_tbl) {
        if (tone_tbl->ne[0] < C) tone_tbl = ggml_transpose(ctx0, tone_tbl);
        tone_ids_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, T);
        ggml_set_input(tone_ids_t);
        struct ggml_tensor *tone_emb = ggml_get_rows(ctx0, tone_tbl, tone_ids_t);
        step_tone = ggml_add(ctx0, step_emb, tone_emb);
        ggml_set_name(step_tone, "s2_tone");
        ggml_set_output(step_tone);
      }

      struct ggml_tensor *step_lang = step_tone;
      struct ggml_tensor *lang_tbl = nullptr;
      auto it_lang = w.find("text_encoder.language_emb.weight");
      if (it_lang != w.end()) lang_tbl = it_lang->second;
      if (lang_tbl) {
        if (lang_tbl->ne[0] < C) lang_tbl = ggml_transpose(ctx0, lang_tbl);
        lang_id_t = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1);
        ggml_set_input(lang_id_t);
        struct ggml_tensor *lang_one = ggml_get_rows(ctx0, lang_tbl, lang_id_t);
        step_lang = ggml_add(ctx0, step_tone, lang_one);
        ggml_set_name(step_lang, "s3_lang");
        ggml_set_output(step_lang);
      }

      struct ggml_tensor *step_scale = ggml_scale(ctx0, step_lang, sqrtf((float)C));
      ggml_set_name(step_scale, "s4_scale");
      ggml_set_output(step_scale);

      pos_enc_t = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
      ggml_set_input(pos_enc_t);
      struct ggml_tensor *step_pe = ggml_add(ctx0, step_scale, pos_enc_t);
      ggml_set_name(step_pe, "s5_pe");
      ggml_set_output(step_pe);

      ggml_build_forward_expand(gf, step_pe);

      ggml_backend_sched_reset(sched);
      if (!ggml_backend_sched_alloc_graph(sched, gf)) { ggml_free(ctx0); return false; }

      // Upload all inputs
      std::vector<int32_t> ids_data = state.phoneme_ids;
      ggml_backend_tensor_set(phoneme_ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));

      if (tone_ids_t) {
        std::vector<int32_t> tone_data = state.tone_ids;
        if (tone_data.empty()) tone_data.resize(T, 0);
        ggml_backend_tensor_set(tone_ids_t, tone_data.data(), 0, tone_data.size()*sizeof(int32_t));
        std::string ts; for(size_t ti=0; ti<tone_data.size() && ti<12; ti++) { if(ti>0)ts+=","; ts+=std::to_string(tone_data[ti]); }
        RS_LOG_INFO("OpenVoice2: tone_ids = [%s]", ts.c_str());
      }

      if (lang_id_t) {
        ggml_backend_tensor_set(lang_id_t, &state.language_id, 0, sizeof(int32_t));
        RS_LOG_INFO("OpenVoice2: lang_id = %d", state.language_id);
      }

      if (pos_enc_t) {
        std::vector<float> pos_data(C * T, 0.0f);
        for (int p = 0; p < T; p++) {
          for (int i = 0; i < C / 2; i++) {
            float freq = powf(10000.0f, -2.0f * (float)i / (float)C);
            pos_data[p * C + (2 * i)]     = sinf((float)p * freq);
            pos_data[p * C + (2 * i + 1)] = cosf((float)p * freq);
          }
        }
        ggml_backend_tensor_set(pos_enc_t, pos_data.data(), 0, pos_data.size()*sizeof(float));
        RS_LOG_INFO("OpenVoice2: pos_enc uploaded first5=%.4f,%.4f,%.4f,%.4f,%.4f",
                    pos_data[0],pos_data[1],pos_data[2],pos_data[3],pos_data[4]);
      }

      ggml_backend_sched_graph_compute(sched, gf);

      // Read back each step
      auto read_and_log = [](struct ggml_tensor *t, const char *label) {
        int n0 = (int)t->ne[0], n1 = (int)t->ne[1];
        std::vector<float> out(n0 * n1);
        ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
        float omin=1e9, omax=-1e9; double osum=0;
        for (size_t i=0; i<out.size(); i++) { if(out[i]<omin) omin=out[i]; if(out[i]>omax) omax=out[i]; osum+=out[i]; }
        RS_LOG_INFO("OpenVoice2: %s [%d,%d] -> [%.4f..%.4f] mean=%.4f first5=%.4f,%.4f,%.4f,%.4f,%.4f",
                    label, n0, n1, omin, omax, (float)(osum/out.size()), out[0],out[1],out[2],out[3],out[4]);
      };

      read_and_log(step_emb, "s1_emb");
      if (step_tone != step_emb) read_and_log(step_tone, "s2_tone");
      if (step_lang != step_tone) read_and_log(step_lang, "s3_lang");
      read_and_log(step_scale, "s4_scale");
      read_and_log(step_pe, "s5_pe");

      ggml_free(ctx0);
      return true;
    }

    int max_layer = DEBUG_LAYERS;
    if (max_layer > n_layers) max_layer = n_layers;

    for (int layer = 0; layer < max_layer; layer++) {
      std::string prefix = "text_encoder.encoder.layers." + std::to_string(layer);
      std::string attn_pref = "text_encoder.encoder.attn_layers." + std::to_string(layer);
      std::string nl1 = "norm_layers_1." + std::to_string(layer);
      std::string nl2 = "norm_layers_2." + std::to_string(layer);
      std::string ffn  = "ffn_layers." + std::to_string(layer);

      struct ggml_tensor *q_w = nullptr, *q_g = nullptr, *q_b = nullptr;
      struct ggml_tensor *k_w = nullptr, *k_g = nullptr, *k_b = nullptr;
      struct ggml_tensor *v_w = nullptr, *v_g = nullptr, *v_b = nullptr;
      struct ggml_tensor *o_w = nullptr, *o_g = nullptr, *o_b = nullptr;
      struct ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
      struct ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
      struct ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
      struct ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
      struct ggml_tensor *emb_rel_k = nullptr;
      struct ggml_tensor *emb_rel_v = nullptr;

      for (auto& [name, t] : w) {
        if (name.find(attn_pref + ".conv_q.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      q_w = t;
          else if (name.find("weight_g") != std::string::npos) q_g = t;
          else if (name.find("weight") != std::string::npos)   q_w = t;
          else if (name.find("bias") != std::string::npos)     q_b = t;
        } else if (name.find(prefix + ".self_attn.q_proj.") != std::string::npos) { q_w = t;
        } else if (name.find(attn_pref + ".conv_k.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      k_w = t;
          else if (name.find("weight_g") != std::string::npos) k_g = t;
          else if (name.find("weight") != std::string::npos)   k_w = t;
          else if (name.find("bias") != std::string::npos)     k_b = t;
        } else if (name.find(prefix + ".self_attn.k_proj.") != std::string::npos) { k_w = t;
        } else if (name.find(attn_pref + ".conv_v.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      v_w = t;
          else if (name.find("weight_g") != std::string::npos) v_g = t;
          else if (name.find("weight") != std::string::npos)   v_w = t;
          else if (name.find("bias") != std::string::npos)     v_b = t;
        } else if (name.find(prefix + ".self_attn.v_proj.") != std::string::npos) { v_w = t;
        } else if (name.find(attn_pref + ".conv_o.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      o_w = t;
          else if (name.find("weight_g") != std::string::npos) o_g = t;
          else if (name.find("weight") != std::string::npos)   o_w = t;
          else if (name.find("bias") != std::string::npos)     o_b = t;
        } else if (name.find(prefix + ".self_attn.o_proj.") != std::string::npos) { o_w = t;
        } else if (name.find(attn_pref + ".emb_rel_k") != std::string::npos) {
          emb_rel_k = t;
        } else if (name.find(attn_pref + ".emb_rel_v") != std::string::npos) {
          emb_rel_v = t;
        } else if (name.find(prefix + ".norm1.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) norm1_w = t; else norm1_b = t;
        } else if (name.find(prefix + ".norm2.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) norm2_w = t; else norm2_b = t;
        } else if (name.find(prefix + ".conv1.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) conv1_w = t; else conv1_b = t;
        } else if (name.find(prefix + ".conv2.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) conv2_w = t; else conv2_b = t;
        } else if (name.find(nl1 + ".") != std::string::npos) {
          if (name.find("gamma") != std::string::npos) norm1_w = t; else norm1_b = t;
        } else if (name.find(nl2 + ".") != std::string::npos) {
          if (name.find("gamma") != std::string::npos) norm2_w = t; else norm2_b = t;
        } else if (name.find(ffn + ".conv_1.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) conv1_w = t; else conv1_b = t;
        } else if (name.find(ffn + ".conv_2.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) conv2_w = t; else conv2_b = t;
        }
      }

      if (q_w && q_g) q_w = apply_weight_norm(ctx0, q_w, q_g);
      if (k_w && k_g) k_w = apply_weight_norm(ctx0, k_w, k_g);
      if (v_w && v_g) v_w = apply_weight_norm(ctx0, v_w, v_g);
      if (o_w && o_g) o_w = apply_weight_norm(ctx0, o_w, o_g);

      struct ggml_tensor *residual = cur;
      if (norm1_w && q_w && k_w && v_w && o_w) {
        cur = layer_norm(ctx0, cur, norm1_w, norm1_b, 1e-5f);
        struct ggml_tensor *attn_out = multi_head_attention(
            ctx0, cur, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
            emb_rel_k, nullptr, n_heads, head_dim, T);
        cur = ggml_add(ctx0, attn_out, residual);
      }

      residual = cur;
      if (conv1_w && conv2_w) {
        if (norm2_w) cur = layer_norm(ctx0, cur, norm2_w, norm2_b, 1e-5f);
        int k_size = conv1_w->ne[0] / C;
        if (k_size <= 0) k_size = 3;
        if (conv1_w->ne[1] >= C) {
          cur = conv1d_im2col(ctx0, cur, conv1_w, conv1_b, k_size, k_size/2, C, conv1_w->ne[2]);
          cur = ggml_relu(ctx0, cur);
          k_size = conv2_w->ne[0] / conv2_w->ne[1];
          if (k_size <= 0) k_size = 3;
          cur = conv1d_im2col(ctx0, cur, conv2_w, conv2_b, k_size, k_size/2, conv2_w->ne[1], conv2_w->ne[2]);
        } else {
          cur = ggml_mul_mat(ctx0, conv1_w, cur);
          if (conv1_b) cur = ggml_add(ctx0, cur, conv1_b);
          cur = ggml_relu(ctx0, cur);
          cur = ggml_mul_mat(ctx0, conv2_w, cur);
          if (conv2_b) cur = ggml_add(ctx0, cur, conv2_b);
        }
        cur = ggml_add(ctx0, cur, residual);
      }
    }

    // Set output and execute
    ggml_set_name(cur, ("debug_layer" + std::to_string(max_layer)).c_str());
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    std::vector<int32_t> ids_data = state.phoneme_ids;
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
      RS_LOG_ERR("OpenVoice2: debug graph alloc failed");
      ggml_free(ctx0);
      return false;
    }
    ggml_backend_tensor_set(phoneme_ids, ids_data.data(), 0, ids_data.size()*sizeof(int32_t));
    ggml_backend_tensor_set(pos_enc, pos_data.data(), 0, pos_data.size()*sizeof(float));
    flush_pending_inputs();
    ggml_backend_sched_graph_compute(sched, gf);

    std::vector<float> out_val(C * T);
    ggml_backend_tensor_get(cur, out_val.data(), 0, out_val.size()*sizeof(float));

    float omin=1e9, omax=-1e9; double osum=0;
    for (int i=0; i<C*T; i++) { if (out_val[i]<omin) omin=out_val[i]; if (out_val[i]>omax) omax=out_val[i]; osum+=out_val[i]; }
    RS_LOG_INFO("OpenVoice2: DEBUG(L%d) -> [%.4f..%.4f] mean=%.4f first5=[%.4f,%.4f,%.4f,%.4f,%.4f]",
                max_layer, omin, omax, (float)(osum/(C*T)),
                out_val[0], out_val[1], out_val[2], out_val[3], out_val[4]);

    // PyTorch per-layer reference values for "你好世界"
    const char* pt_ref[] = {
      "L0: [-44.62,57.43] mean=0.35",
      "L1: [-106.58,92.90] mean=0.13",
      "L2: [-160.71,118.94] mean=-0.13",
      "L3: [-162.95,131.77] mean=0.22",
      "L4: [-179.99,145.25] mean=-0.12",
      "L5: [-180.83,145.48] mean=-0.40",
    };
    if (max_layer <= 6) RS_LOG_INFO("OpenVoice2: PyTorch ref: %s", pt_ref[max_layer > 0 ? max_layer-1 : 0]);

    ggml_free(ctx0);
    return true;
  }
  #undef DEBUG_LAYERS
  #endif

  // Debug: checkpoint after embeddings
  std::vector<struct ggml_tensor*> layer_checkpoints;

  // --- N Transformer Layers ---
  for (int layer = 0; layer < n_layers; layer++) {
    // Build weight name prefixes for this layer
    std::string prefix = "text_encoder.encoder.layers." + std::to_string(layer);
    std::string attn_pref = "text_encoder.encoder.attn_layers." + std::to_string(layer);

    struct ggml_tensor *residual = cur;

    // --- Self-Attention ---
    // Try attn_layers naming first (MeloTTS's relative position transformer),
    // fallback to standard transformer naming
    struct ggml_tensor *q_w = nullptr, *q_g = nullptr, *q_b = nullptr;
    struct ggml_tensor *k_w = nullptr, *k_g = nullptr, *k_b = nullptr;
    struct ggml_tensor *v_w = nullptr, *v_g = nullptr, *v_b = nullptr;
    struct ggml_tensor *o_w = nullptr, *o_g = nullptr, *o_b = nullptr;
    struct ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
    struct ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr;
    struct ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
    struct ggml_tensor *conv2_w = nullptr, *conv2_b = nullptr;
    struct ggml_tensor *emb_rel_k = nullptr;
    struct ggml_tensor *emb_rel_v = nullptr;

    // Look up weights by common naming conventions
    // Chinese MeloTTS: norm_layers_1.<N>.gamma/beta, ffn_layers.<N>.conv_1.*
    // weight_norm: weight_v (direction) + weight_g (scale) → effective weight
    std::string nl1 = "norm_layers_1." + std::to_string(layer);
    std::string nl2 = "norm_layers_2." + std::to_string(layer);
    std::string ffn  = "ffn_layers." + std::to_string(layer);
    for (auto& [name, t] : w) {
      if (name.find(attn_pref + ".conv_q.") != std::string::npos) {
        if (name.find("weight_v") != std::string::npos)      q_w = t;
        else if (name.find("weight_g") != std::string::npos) q_g = t;
        else if (name.find("weight") != std::string::npos)   q_w = t;  // plain weight
        else if (name.find("bias") != std::string::npos)     q_b = t;
      } else if (name.find(prefix + ".self_attn.q_proj.") != std::string::npos) { q_w = t;
      } else if (name.find(attn_pref + ".conv_k.") != std::string::npos) {
        if (name.find("weight_v") != std::string::npos)      k_w = t;
        else if (name.find("weight_g") != std::string::npos) k_g = t;
        else if (name.find("weight") != std::string::npos)   k_w = t;  // plain weight
        else if (name.find("bias") != std::string::npos)     k_b = t;
      } else if (name.find(prefix + ".self_attn.k_proj.") != std::string::npos) { k_w = t;
      } else if (name.find(attn_pref + ".conv_v.") != std::string::npos) {
        if (name.find("weight_v") != std::string::npos)      v_w = t;
        else if (name.find("weight_g") != std::string::npos) v_g = t;
        else if (name.find("weight") != std::string::npos)   v_w = t;  // plain weight
        else if (name.find("bias") != std::string::npos)     v_b = t;
      } else if (name.find(prefix + ".self_attn.v_proj.") != std::string::npos) { v_w = t;
      } else if (name.find(attn_pref + ".conv_o.") != std::string::npos) {
        if (name.find("weight_v") != std::string::npos)      o_w = t;
        else if (name.find("weight_g") != std::string::npos) o_g = t;
        else if (name.find("weight") != std::string::npos)   o_w = t;  // plain weight
        else if (name.find("bias") != std::string::npos)     o_b = t;
      } else if (name.find(prefix + ".self_attn.o_proj.") != std::string::npos) { o_w = t;
      } else if (name.find(attn_pref + ".emb_rel_k") != std::string::npos) {
        emb_rel_k = t;
      } else if (name.find(attn_pref + ".emb_rel_v") != std::string::npos) {
        emb_rel_v = t;
      } else if (name.find(prefix + ".norm1.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) norm1_w = t; else norm1_b = t;
      } else if (name.find(prefix + ".norm2.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) norm2_w = t; else norm2_b = t;
      } else if (name.find(prefix + ".conv1.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv1_w = t; else conv1_b = t;
      } else if (name.find(prefix + ".conv2.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv2_w = t; else conv2_b = t;
      } else if (name.find(nl1 + ".") != std::string::npos) {
        if (name.find("gamma") != std::string::npos) norm1_w = t; else norm1_b = t;
      } else if (name.find(nl2 + ".") != std::string::npos) {
        if (name.find("gamma") != std::string::npos) norm2_w = t; else norm2_b = t;
      } else if (name.find(ffn + ".conv_1.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv1_w = t; else conv1_b = t;
      } else if (name.find(ffn + ".conv_2.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) conv2_w = t; else conv2_b = t;
      }
    }

    // Apply weight_norm: reconstruct effective weight = g * v / ||v||
    if (q_w && q_g) q_w = apply_weight_norm(ctx0, q_w, q_g);
    if (k_w && k_g) k_w = apply_weight_norm(ctx0, k_w, k_g);
    if (v_w && v_g) v_w = apply_weight_norm(ctx0, v_w, v_g);
    if (o_w && o_g) o_w = apply_weight_norm(ctx0, o_w, o_g);

    // === POST-NORM architecture (MeloTTS/VITS standard) ===
    // Each sub-layer: x = Norm(x + Sublayer(x))
    // This keeps values bounded through normalization at every sub-layer output.

    // --- Self-Attention block ---
    if (norm1_w && q_w && k_w && v_w && o_w) {
      // Quick sanity: read raw norm1_w values (only layer 0)
      if (layer == 0) {
        std::vector<float> nw(10);
        ggml_backend_tensor_get(norm1_w, nw.data(), 0, 10*sizeof(float));
        RS_LOG_INFO("OpenVoice2: layer0 norm1_w first10=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
                    nw[0],nw[1],nw[2],nw[3],nw[4],nw[5],nw[6],nw[7],nw[8],nw[9]);
      }
      struct ggml_tensor *attn_out = multi_head_attention(
          ctx0, cur, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b,
          emb_rel_k, nullptr, n_heads, head_dim, T);
      cur = ggml_add(ctx0, cur, attn_out);
      cur = layer_norm(ctx0, cur, norm1_w, norm1_b, 1e-5f);
      // Debug: checkpoint after attention + post-norm
      {
        ggml_set_name(cur, ("layer_" + std::to_string(layer) + "_attn_norm").c_str());
        layer_checkpoints.push_back(cur);
      }
    } else {
      RS_LOG_WARN("OpenVoice2: text_encoder layer %d missing attention weights, skipping", layer);
    }

    // --- FFN block (Conv1D) ---
    if (conv1_w && conv2_w) {
      struct ggml_tensor *ffn_in = cur;
      int k_size = conv1_w->ne[0] / C;  // infer kernel size from weight
      if (k_size <= 0) k_size = 3;
      if (conv1_w->ne[1] >= C) {
        // Debug: print weight shapes and values for layer 0
        if (layer == 0) {
          RS_LOG_INFO("OpenVoice2: L0 FFN conv1_w shape=%lldx%lldx%lld (k_size=%d, C=%d), conv2_w shape=%lldx%lldx%lld",
              (long long)conv1_w->ne[0], (long long)conv1_w->ne[1], (long long)conv1_w->ne[2], k_size, C,
              (long long)conv2_w->ne[0], (long long)conv2_w->ne[1], (long long)conv2_w->ne[2]);
          // Print first 10 values of conv1_w
          int nw = std::min(10, (int)(conv1_w->ne[0] * conv1_w->ne[1] * conv1_w->ne[2]));
          std::vector<float> c1w(nw);
          ggml_backend_tensor_get(conv1_w, c1w.data(), 0, nw * sizeof(float));
          RS_LOG_INFO("OpenVoice2: L0 conv1_w first10=%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
              c1w[0],c1w[1],c1w[2],c1w[3],c1w[4],c1w[5],c1w[6],c1w[7],c1w[8],c1w[9]);
          if (conv1_b) {
            std::vector<float> c1b(std::min(10, (int)conv1_b->ne[0]));
            ggml_backend_tensor_get(conv1_b, c1b.data(), 0, c1b.size() * sizeof(float));
            RS_LOG_INFO("OpenVoice2: L0 conv1_b first10=%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
                c1b[0],c1b[1],c1b[2],c1b[3],c1b[4],c1b[5],c1b[6],c1b[7],c1b[8],c1b[9]);
          }
        }
        // Weight: [kw, in, out] → ne[0]=kw, ne[1]=in, ne[2]=out
        struct ggml_tensor *ffn_h = conv1d_im2col(ctx0, cur, conv1_w, conv1_b, k_size,
                            k_size / 2, C, conv1_w->ne[2]);
        // Debug layer 0 FFN intermediate
        if (layer == 0) {
          RS_LOG_INFO("OpenVoice2: L0 ffn_h(pre-relu) shape=%lldx%lld",
              (long long)ffn_h->ne[0], (long long)ffn_h->ne[1]);
        }
        ffn_h = ggml_relu(ctx0, ffn_h);

        k_size = conv2_w->ne[0] / conv2_w->ne[1];
        if (k_size <= 0) k_size = 3;
        struct ggml_tensor *ffn_out = conv1d_im2col(ctx0, ffn_h, conv2_w, conv2_b, k_size,
                            k_size / 2, conv2_w->ne[1], conv2_w->ne[2]);
        // Debug layer 0 FFN output
        if (layer == 0) {
          RS_LOG_INFO("OpenVoice2: L0 ffn_out shape=%lldx%lld",
              (long long)ffn_out->ne[0], (long long)ffn_out->ne[1]);
          ggml_set_name(ffn_out, "layer_0_ffn_raw");
        }
        cur = ggml_add(ctx0, ffn_in, ffn_out);
      } else {
        // Fallback: use mul_mat for linear layers
        struct ggml_tensor *ffn_h = ggml_mul_mat(ctx0, conv1_w, cur);
        if (conv1_b) ffn_h = ggml_add(ctx0, ffn_h, conv1_b);
        ffn_h = ggml_relu(ctx0, ffn_h);
        struct ggml_tensor *ffn_out = ggml_mul_mat(ctx0, conv2_w, ffn_h);
        if (conv2_b) ffn_out = ggml_add(ctx0, ffn_out, conv2_b);
        cur = ggml_add(ctx0, ffn_in, ffn_out);
      }

      if (norm2_w) {
        cur = layer_norm(ctx0, cur, norm2_w, norm2_b, 1e-5f);
      }
    } else {
      RS_LOG_WARN("OpenVoice2: text_encoder layer %d missing FFN weights, skipping", layer);
    }

    // Save checkpoint for this layer (after FFN + norm)
    ggml_set_name(cur, ("layer_" + std::to_string(layer) + "_ffn_out").c_str());
    layer_checkpoints.push_back(cur);
  }

  // --- Final LayerNorm ---
  struct ggml_tensor *final_norm_w = nullptr, *final_norm_b = nullptr;
  for (auto& [name, t] : w) {
    if (name.find("after_norm.") != std::string::npos ||
        name.find("norm_post.") != std::string::npos) {
      if (name.find("weight") != std::string::npos || name.find("gamma") != std::string::npos)
        final_norm_w = t;
      else final_norm_b = t;
    }
  }
  if (final_norm_w) {
    cur = layer_norm(ctx0, cur, final_norm_w, final_norm_b, 1e-5f);
  }

  // --- Output Projection: proj(hidden) → [m_p | logs_p] ---
  // Conv1d(hidden_channels, out_channels*2, 1): projects 192 → 384
  // Split: first 192 = m_p (mean), last 192 = logs_p (log-scale)
  struct ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
  for (auto& [name, t] : w) {
    if (name.find("proj.") != std::string::npos &&
        name.find("bert_proj") == std::string::npos &&
        name.find("ja_bert") == std::string::npos) {
      if (name.find("weight") != std::string::npos) proj_w = t;
      else if (name.find("bias") != std::string::npos) proj_b = t;
    }
  }

  struct ggml_tensor *stats = nullptr;
  if (proj_w) {
    // Debug: verify proj weight values
    {
      int nproj = (int)(proj_w->ne[0] * proj_w->ne[1] * proj_w->ne[2]);
      int nread = std::min(20, nproj);
      std::vector<float> pw(nread);
      ggml_backend_tensor_get(proj_w, pw.data(), 0, nread*sizeof(float));
      float pmn=1e9,pmx=-1e9; double psum=0;
      for(auto v:pw){if(v<pmn)pmn=v;if(v>pmx)pmx=v;psum+=v;}
      RS_LOG_INFO("OpenVoice2: proj_w %lldx%lldx%lld first20=[%.4f..%.4f] mean=%.4f",
          (long long)proj_w->ne[0],(long long)proj_w->ne[1],(long long)proj_w->ne[2],pmn,pmx,(float)(psum/nread));
      if (proj_b) {
        int nbref = std::min(10, (int)proj_b->ne[0]);
        std::vector<float> pb(nbref);
        ggml_backend_tensor_get(proj_b, pb.data(), 0, nbref*sizeof(float));
        RS_LOG_INFO("OpenVoice2: proj_b first10=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
            pb[0],pb[1],pb[2],pb[3],pb[4],pb[5],pb[6],pb[7],pb[8],pb[9]);
      }
    }
    cur = ggml_cont(ctx0, cur);
    // Use conv1d_im2col (same as FFN convs, already verified working)
    stats = conv1d_im2col(ctx0, cur, proj_w, proj_b, 1, 0, C, C * 2);
    ggml_set_name(stats, "proj_stats");
  }

  ggml_set_name(cur, "text_encoder_out");

  // Build graph: mark all checkpoints, emb_out, text_encoder_out, and proj_stats
  // as outputs so their buffers are preserved after graph computation.
  // Without this, intermediate tensors get buffer-reuse'd and show garbage.
  for (auto* ckpt : layer_checkpoints) {
    ggml_set_output(ckpt);
    ggml_build_forward_expand(gf, ckpt);
  }
  ggml_set_output(emb_out_tensor);
  ggml_build_forward_expand(gf, emb_out_tensor);
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);
  struct ggml_tensor *graph_out = stats ? stats : cur;
  if (stats) {
    ggml_set_output(stats);
    ggml_build_forward_expand(gf, stats);
  }

  // --- Execute ---
  std::vector<int32_t> ids_data = state.phoneme_ids;

  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: TextEncoder graph allocation failed");
    ggml_free(ctx0);
    return false;
  }

  // Upload input data after graph allocation (backend buffers are now assigned)
  ggml_backend_tensor_set(phoneme_ids, ids_data.data(), 0,
                          ids_data.size() * sizeof(int32_t));

  // Flush relative-position index/mask tensors registered by multi_head_attention
  flush_pending_inputs();

  ggml_backend_sched_graph_compute(sched, gf);

  // Debug: read embedding output values from the saved tensor (before layers)
  {
    std::vector<float> emb_data(C * T);
    ggml_backend_tensor_get(emb_out_tensor, emb_data.data(), 0, emb_data.size() * sizeof(float));
    float emin=1e9, emax=-1e9; double esum=0;
    for (size_t i=0; i<emb_data.size(); i++) {
      if (emb_data[i]<emin) emin=emb_data[i];
      if (emb_data[i]>emax) emax=emb_data[i];
      esum+=emb_data[i];
    }
    RS_LOG_INFO("OpenVoice2: EMB(no layers) -> [%.4f..%.4f] mean=%.4f first5=[%.4f,%.4f,%.4f,%.4f,%.4f]",
                emin, emax, (float)(esum/emb_data.size()),
                emb_data[0], emb_data[1], emb_data[2], emb_data[3], emb_data[4]);
  }

  // --- Per-layer diagnostics ---
  for (size_t li = 0; li < layer_checkpoints.size(); li++) {
    auto* ckpt = layer_checkpoints[li];
    int ckpt_C = (int)ckpt->ne[0];
    int ckpt_T = (int)ckpt->ne[1];
    std::vector<float> ckpt_data(ckpt_C * ckpt_T);
    ggml_backend_tensor_get(ckpt, ckpt_data.data(), 0,
                            ckpt_data.size() * sizeof(float));
    float cmin = 1e9, cmax = -1e9;
    double csum = 0;
    int nzero = 0;
    for (size_t i = 0; i < ckpt_data.size(); i++) {
      float v = ckpt_data[i];
      if (v < cmin) cmin = v;
      if (v > cmax) cmax = v;
      csum += v;
      if (v == 0.0f) nzero++;
    }
    const char* lbl = ckpt->name ? ckpt->name : "?";
    RS_LOG_INFO("OpenVoice2: ckpt[%zu] %s -> [%.4f..%.4f] mean=%.4f nonzero=%d/%d",
                li, lbl, cmin, cmax, (float)(csum / ckpt_data.size()),
                (int)(ckpt_data.size() - nzero), (int)ckpt_data.size());
  }

  // --- Read outputs ---
  state.encoder_hidden.resize(C * T);
  ggml_backend_tensor_get(cur, state.encoder_hidden.data(), 0,
                          state.encoder_hidden.size() * sizeof(float));
  state.encoder_T = T;

  // Read m_p and logs_p from proj output
  state.m_p.resize(C * T);
  state.logs_p.resize(C * T);
  if (stats) {
    std::vector<float> stats_data(C * 2 * T);
    ggml_backend_tensor_get(stats, stats_data.data(), 0,
                            stats_data.size() * sizeof(float));
    // conv1d_im2col output layout: [ne0=C*2, ne1=T] → data[ch + t * (C*2)]
    // Target: m_p/logs_p are [C, T] → data[ch + t * C]
    for (int t = 0; t < T; t++) {
      for (int c = 0; c < C; c++) {
        state.m_p[c + t * C]    = stats_data[c + t * (C * 2)];
        state.logs_p[c + t * C] = stats_data[(c + C) + t * (C * 2)];
      }
    }

    float mp_min = 1e9, mp_max = -1e9, lp_min = 1e9, lp_max = -1e9;
    for (int i = 0; i < C * T; i++) {
      if (state.m_p[i] < mp_min) mp_min = state.m_p[i];
      if (state.m_p[i] > mp_max) mp_max = state.m_p[i];
      if (state.logs_p[i] < lp_min) lp_min = state.logs_p[i];
      if (state.logs_p[i] > lp_max) lp_max = state.logs_p[i];
    }
    RS_LOG_INFO("OpenVoice2: proj m_p [%.4f..%.4f] logs_p [%.4f..%.4f]",
                mp_min, mp_max, lp_min, lp_max);
    // Print first 5 values for comparison with PyTorch
    if (T > 0 && C >= 5) {
      RS_LOG_INFO("OpenVoice2: first 5 hidden: %.4f, %.4f, %.4f, %.4f, %.4f",
                  state.encoder_hidden[0], state.encoder_hidden[1],
                  state.encoder_hidden[2], state.encoder_hidden[3],
                  state.encoder_hidden[4]);
      RS_LOG_INFO("OpenVoice2: first 5 m_p: %.4f, %.4f, %.4f, %.4f, %.4f",
                  state.m_p[0], state.m_p[1], state.m_p[2],
                  state.m_p[3], state.m_p[4]);
      RS_LOG_INFO("OpenVoice2: first 5 logs_p: %.4f, %.4f, %.4f, %.4f, %.4f",
                  state.logs_p[0], state.logs_p[1], state.logs_p[2],
                  state.logs_p[3], state.logs_p[4]);
    }
  } else {
    // Fallback: if no proj weight, set m_p = hidden, logs_p = 0
    std::copy(state.encoder_hidden.begin(), state.encoder_hidden.end(),
              state.m_p.begin());
    std::fill(state.logs_p.begin(), state.logs_p.end(), 0.0f);
  }

  {
    float hmin = 1e9, hmax = -1e9;
    double hsum = 0;
    for (int i = 0; i < C * T; i++) {
      float v = state.encoder_hidden[i];
      if (v < hmin) hmin = v;
      if (v > hmax) hmax = v;
      hsum += v;
    }
    RS_LOG_INFO("OpenVoice2: TextEncoder hidden values [%.4f..%.4f] mean=%.4f (%d values)",
                hmin, hmax, (float)(hsum / (C * T)), C * T);
  }

  ggml_free(ctx0);
  RS_LOG_INFO("OpenVoice2: TextEncoder -> [%d, %d]", C, T);
  return true;
}

// =====================================================================
// Sub-graph: Duration Predictor
//
// Architecture: Conv1D → LayerNorm → ReLU → Conv1D → LayerNorm → ReLU → Linear → exp
// =====================================================================

bool OpenVoice2Model::RunDurationPredictor(OpenVoice2State& state,
                                            ggml_backend_sched_t sched) {
  auto& w = weights_.duration_predictor;
  if (w.empty()) {
    RS_LOG_WARN("OpenVoice2: no duration_predictor weights, using default durations");
    // Fallback: assign default durations
    int T = state.encoder_T;
    state.durations.resize(T);
    state.total_mel_frames = 0;
    for (int t = 0; t < T; t++) {
      state.durations[t] = 5;
      state.total_mel_frames += 5;
    }
    return true;
  }

  int T = state.encoder_T;
  int C = hparams_.hidden_channels;

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for DurationPredictor");
    return false;
  }

  // Input: encoder hidden [C, T]
  struct ggml_tensor *hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T);
  ggml_set_name(hidden, "encoder_hidden");
  ggml_set_input(hidden);

  struct ggml_tensor *cur = hidden;

  // Find weights
  struct ggml_tensor *conv0_w = nullptr, *conv0_b = nullptr;
  struct ggml_tensor *conv1_w = nullptr, *conv1_b = nullptr;
  struct ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
  struct ggml_tensor *norm0_w = nullptr, *norm0_b = nullptr;
  struct ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr;
  struct ggml_tensor *cond_w = nullptr, *cond_b = nullptr;

  for (auto& [name, t] : w) {
    // Chinese MeloTTS: conv_1/conv_2, norm_1/norm_2
    // English MeloTTS: conv.0/conv.1, norm.0/norm.1
    if (name.find("conv_1.") != std::string::npos ||
        name.find("conv.0.") != std::string::npos) {
      if (name.find("weight") != std::string::npos || name.find("gamma") == std::string::npos)
        conv0_w = t; else conv0_b = t;
    } else if (name.find("conv_2.") != std::string::npos ||
               name.find("conv.1.") != std::string::npos) {
      if (name.find("weight") != std::string::npos || name.find("gamma") == std::string::npos)
        conv1_w = t; else conv1_b = t;
    } else if (name.find("norm_1.") != std::string::npos ||
               name.find("norm.0.") != std::string::npos) {
      if (name.find("gamma") != std::string::npos || name.find("weight") != std::string::npos)
        norm0_w = t; else norm0_b = t;
    } else if (name.find("norm_2.") != std::string::npos ||
               name.find("norm.1.") != std::string::npos) {
      if (name.find("gamma") != std::string::npos || name.find("weight") != std::string::npos)
        norm1_w = t; else norm1_b = t;
    } else if (name.find("proj.") != std::string::npos) {
      if (name.find("weight") != std::string::npos) proj_w = t; else proj_b = t;
    } else if (name.find("cond.") != std::string::npos) {
      if (name.find("weight") != std::string::npos) cond_w = t; else cond_b = t;
    }
  }

  if (!conv0_w || !proj_w) {
    RS_LOG_WARN("OpenVoice2: duration predictor missing key weights, using default durations");
    ggml_free(ctx0);
    int T2 = state.encoder_T;
    state.durations.resize(T2);
    state.total_mel_frames = 0;
    for (int t = 0; t < T2; t++) {
      state.durations[t] = 5;
      state.total_mel_frames += 5;
    }
    return true;
  }

  // Conv block 0
  int k_size = conv0_w->ne[0] / C;
  if (k_size <= 0) k_size = 3;
  int out_ch0 = static_cast<int>(conv0_w->ne[2]);
  if (out_ch0 <= 0) out_ch0 = static_cast<int>(conv0_w->ne[1]);
  if (conv0_w->ne[0] >= 3 && conv0_w->ne[1] >= C) {
    cur = conv1d_im2col(ctx0, cur, conv0_w, conv0_b, k_size, k_size / 2, C, out_ch0);
  } else {
    cur = ggml_mul_mat(ctx0, conv0_w, cur);
    if (conv0_b) cur = ggml_add(ctx0, cur, conv0_b);
  }


    // Cond: PyTorch forward(x, x_mask, g=None) only applies cond(g) when
    // a speaker embedding is provided. TTS inference has none, so skip.

    int cur_ch = cur->ne[0];
    cur = ggml_relu(ctx0, cur);
    if (norm0_w && norm0_w->ne[0] == cur_ch) {
        cur = layer_norm(ctx0, cur, norm0_w, norm0_b, 1e-5f);
    }

  // Conv block 1
  if (conv1_w) {
    int cur_ch1 = static_cast<int>(cur->ne[0]);
    int out_ch1 = static_cast<int>(conv1_w->ne[2]);
    if (out_ch1 <= 0) out_ch1 = static_cast<int>(conv1_w->ne[1]);
    if (conv1_w->ne[0] >= 3 && conv1_w->ne[1] >= cur_ch1) {
      k_size = 3;
      cur = conv1d_im2col(ctx0, cur, conv1_w, conv1_b, k_size, k_size / 2, cur_ch1, out_ch1);
    } else {
      cur = ggml_mul_mat(ctx0, conv1_w, cur);
      if (conv1_b) cur = ggml_add(ctx0, cur, conv1_b);
    }


    cur = ggml_relu(ctx0, cur);
    if (norm1_w && norm1_w->ne[0] == cur->ne[0]) {
        cur = layer_norm(ctx0, cur, norm1_w, norm1_b, 1e-5f);
    }
  }

  // Projection to 1 dimension
  // proj weight may be 3D [1, in, 1] (1x1 conv) → squeeze to 2D [in, 1]
  if (proj_w->ne[0] == 1 && proj_w->ne[2] > 0)
    proj_w = ggml_reshape_2d(ctx0, ggml_cont(ctx0, proj_w), proj_w->ne[1], proj_w->ne[2]);
  cur = ggml_mul_mat(ctx0, proj_w, cur);
  if (proj_b) cur = ggml_add(ctx0, cur, proj_b);

  // Squeeze channel dimension: [1, T] → [T]
  cur = ggml_reshape_1d(ctx0, cur, T);

  // exp() to get positive durations (already the correct scale)
  cur = ggml_exp(ctx0, cur);

  ggml_set_name(cur, "durations");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // --- Execute ---
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: DurationPredictor graph allocation failed");
    ggml_free(ctx0);
    return false;
  }

  ggml_backend_tensor_set(hidden, state.encoder_hidden.data(), 0,
                          state.encoder_hidden.size() * sizeof(float));

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read output ---
  std::vector<float> dur_float(T);
  ggml_backend_tensor_get(cur, dur_float.data(), 0, T * sizeof(float));

  state.durations.resize(T);
  state.total_mel_frames = 0;
  float dur_min = 1e9, dur_max = -1e9;
  for (int t = 0; t < T; t++) {
    if (dur_float[t] < dur_min) dur_min = dur_float[t];
    if (dur_float[t] > dur_max) dur_max = dur_float[t];
    int dur = std::max(1, static_cast<int>(std::ceil(dur_float[t])));
    state.durations[t] = dur;
    state.total_mel_frames += dur;
  }
  RS_LOG_INFO("OpenVoice2: raw durations [%.4f..%.4f] (%d values)", dur_min, dur_max, T);

  ggml_free(ctx0);
  RS_LOG_INFO("OpenVoice2: DurationPredictor -> %d total mel frames", state.total_mel_frames);
  return true;
}

// =====================================================================
// Sub-graph: Flow Decoder
//
// Architecture: Expand hidden by durations → N coupling blocks with
// WaveNet-style affine transforms
// =====================================================================

bool OpenVoice2Model::RunFlowDecoder(OpenVoice2State& state,
                                      ggml_backend_sched_t sched) {
  auto& w = weights_.flow_decoder;
  int n_mels = hparams_.n_mels;
  int T_mel = state.total_mel_frames;
  int C = hparams_.hidden_channels;
  int T_txt = state.encoder_T;


  if (T_mel <= 0) {
    RS_LOG_ERR("OpenVoice2: no mel frames to generate");
    return false;
  }

  if (w.empty()) {
    RS_LOG_WARN("OpenVoice2: no flow_decoder weights, generating placeholder mel");
    // Fallback: simple expansion of encoder hidden
    state.mel_spectrogram.resize(n_mels * T_mel, 0.0f);
    int mel_pos = 0;
    for (int t = 0; t < T_txt && mel_pos < T_mel; t++) {
      int dur = state.durations[t];
      for (int d = 0; d < dur && mel_pos < T_mel; d++) {
        for (int m = 0; m < std::min(n_mels, C); m++) {
          state.mel_spectrogram[m + mel_pos * n_mels] =
              state.encoder_hidden[m + t * C];
        }
        mel_pos++;
      }
    }
    return true;
  }

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for FlowDecoder");
    return false;
  }

  g_pending_inputs.clear();

  struct ggml_tensor *cur;

  // --- Load sampled prior z_p (already expanded to mel frames) ---
  struct ggml_tensor *z_p_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, C, T_mel);
  ggml_set_name(z_p_tensor, "z_p");
  ggml_set_input(z_p_tensor);
  cur = z_p_tensor;

  // --- Flow coupling blocks (reverse order during inference) ---
  // Chinese MeloTTS: flows at even indices (0,2,4,6) have pre+enc+post transformer;
  // odd indices (1,3,5,7) are flip-only blocks.
  // English MeloTTS: flows at consecutive indices with in_layers WaveNet.
  // During inference (reverse=True), flows are processed from n_flows-1 down to 0.
  int n_flows = hparams_.n_flow_layers;
  int C_flow = cur->ne[0];  // flow operates on this many channels (hidden_channels)
  int half = C_flow / 2;

  for (int flow = n_flows - 1; flow >= 0; flow--) {
    std::string flow_pref = "flow_decoder.flows." + std::to_string(flow);

    // Check if this flow has weights (coupling) or is flip-only
    bool has_weights = false;
    for (auto& [name, t] : w) {
      if (name.find(flow_pref + ".") == 0) { has_weights = true; break; }
    }

    if (!has_weights) {
      // Flip-only block: swap x_a and x_b
      struct ggml_tensor *x_a = ggml_view_2d(ctx0, cur, half, cur->ne[1],
                                              cur->nb[1], 0);
      struct ggml_tensor *x_b = ggml_view_2d(ctx0, cur, C_flow - half,
                                              cur->ne[1], cur->nb[1],
                                              half * sizeof(float));
      cur = ggml_concat(ctx0, x_b, x_a, 0);
      continue;
    }

    // --- Coupling block ---
    // Find weights: Chinese model uses pre/enc/post, English uses in_layers/WaveNet
    struct ggml_tensor *pre_w = nullptr, *pre_b = nullptr;   // 1x1 pre-conv
    struct ggml_tensor *post_w = nullptr, *post_b = nullptr;  // 1x1 post-conv
    // English model WaveNet weights:
    struct ggml_tensor *in_l0_w = nullptr, *in_l1_w = nullptr, *in_l2_w = nullptr;
    // Chinese model: transformer encoder weights per layer
    struct FlowEncWeights {
      struct ggml_tensor *norm1_w, *norm1_b;
      struct ggml_tensor *norm2_w, *norm2_b;
      struct ggml_tensor *q_w, *q_g, *q_b, *k_w, *k_g, *k_b;
      struct ggml_tensor *v_w, *v_g, *v_b, *o_w, *o_g, *o_b;
      struct ggml_tensor *conv1_w, *conv1_b, *conv2_w, *conv2_b;
      struct ggml_tensor *emb_rel_k = nullptr;
      struct ggml_tensor *emb_rel_v = nullptr;
    };
    std::vector<FlowEncWeights> enc_layers;
    int n_enc_layers = 0;

    for (auto& [name, t] : w) {
      if (name.find(flow_pref + ".") != 0) continue;
      // Chinese model naming
      if (name.find(".pre.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) pre_w = t; else pre_b = t;
      } else if (name.find(".post.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) post_w = t; else post_b = t;
      } else if (name.find(".enc.attn_layers.") != std::string::npos) {
        // Parse layer index: .enc.attn_layers.N.
        size_t p = name.find(".enc.attn_layers.") + 17;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("conv_q.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      ew.q_w = t;
          else if (name.find("weight_g") != std::string::npos) ew.q_g = t;
          else if (name.find("weight") != std::string::npos)   ew.q_w = t;  // plain weight
          else if (name.find("bias") != std::string::npos)     ew.q_b = t;
        } else if (name.find("conv_k.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      ew.k_w = t;
          else if (name.find("weight_g") != std::string::npos) ew.k_g = t;
          else if (name.find("weight") != std::string::npos)   ew.k_w = t;  // plain weight
          else if (name.find("bias") != std::string::npos)     ew.k_b = t;
        } else if (name.find("conv_v.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      ew.v_w = t;
          else if (name.find("weight_g") != std::string::npos) ew.v_g = t;
          else if (name.find("weight") != std::string::npos)   ew.v_w = t;  // plain weight
          else if (name.find("bias") != std::string::npos)     ew.v_b = t;
        } else if (name.find("conv_o.") != std::string::npos) {
          if (name.find("weight_v") != std::string::npos)      ew.o_w = t;
          else if (name.find("weight_g") != std::string::npos) ew.o_g = t;
          else if (name.find("weight") != std::string::npos)   ew.o_w = t;  // plain weight
          else if (name.find("bias") != std::string::npos)     ew.o_b = t;
        } else if (name.find("emb_rel_k") != std::string::npos) {
          ew.emb_rel_k = t;
        } else if (name.find("emb_rel_v") != std::string::npos) {
          ew.emb_rel_v = t;
        }
      } else if (name.find(".enc.norm_layers_1.") != std::string::npos) {
        size_t p = name.find(".enc.norm_layers_1.") + 20;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("gamma") != std::string::npos) ew.norm1_w = t; else ew.norm1_b = t;
      } else if (name.find(".enc.norm_layers_2.") != std::string::npos) {
        size_t p = name.find(".enc.norm_layers_2.") + 20;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("gamma") != std::string::npos) ew.norm2_w = t; else ew.norm2_b = t;
      } else if (name.find(".enc.ffn_layers.") != std::string::npos) {
        size_t p = name.find(".enc.ffn_layers.") + 17;
        int lyr = 0; while (name[p] >= '0' && name[p] <= '9') { lyr = lyr*10+(name[p]-'0'); p++; }
        if (lyr >= static_cast<int>(enc_layers.size())) enc_layers.resize(lyr + 1);
        auto& ew = enc_layers[lyr];
        if (name.find("conv_1.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.conv1_w = t; else ew.conv1_b = t;
        } else if (name.find("conv_2.") != std::string::npos) {
          if (name.find("weight") != std::string::npos) ew.conv2_w = t; else ew.conv2_b = t;
        }
      }
      // English model naming (WaveNet)
      else if (name.find(".in_layers.0.") != std::string::npos) in_l0_w = t;
      else if (name.find(".in_layers.1.") != std::string::npos) in_l1_w = t;
      else if (name.find(".in_layers.2.") != std::string::npos) in_l2_w = t;
    }
    n_enc_layers = static_cast<int>(enc_layers.size());

    // Check if this is a Chinese-style (pre+enc+post) or English-style (WaveNet) flow
    bool is_chinese_flow = (pre_w && post_w && n_enc_layers > 0);

    if (!is_chinese_flow && (!in_l0_w || !in_l2_w)) {
      RS_LOG_WARN("OpenVoice2: flow layer %d missing key weights, skipping", flow);
      continue;
    }

    // Split into x_a (first half) and x_b (second half)
    struct ggml_tensor *x_a = ggml_view_2d(ctx0, cur, half, cur->ne[1],
                                            cur->nb[1], 0);
    struct ggml_tensor *x_b = ggml_view_2d(ctx0, cur, C_flow - half,
                                            cur->ne[1], cur->nb[1],
                                            half * sizeof(float));

    if (is_chinese_flow) {
      // Chinese model: pre → transformer enc → post → additive coupling
      int n_heads = hparams_.n_heads;
      int head_dim = C_flow / n_heads;

      // Pre conv: [half, T] → [C_flow, T]
      struct ggml_tensor *h;
      if (pre_w->ne[1] >= half) {
        h = conv1d_im2col(ctx0, x_a, pre_w, pre_b, 1, 0, half, C_flow);
      } else {
        h = ggml_mul_mat(ctx0, pre_w, x_a);
        if (pre_b) h = ggml_add(ctx0, h, pre_b);
      }

      // Transformer encoder layers (POST-norm, matching PyTorch attentions.Encoder)
      for (int l = 0; l < n_enc_layers; l++) {
        auto& ew = enc_layers[l];
        if (!ew.norm1_w || !ew.q_w) continue;

        // Apply weight_norm: reconstruct effective weight = g * v / ||v||
        struct ggml_tensor *q_eff = ew.q_w, *k_eff = ew.k_w;
        struct ggml_tensor *v_eff = ew.v_w, *o_eff = ew.o_w;
        if (ew.q_w && ew.q_g) q_eff = apply_weight_norm(ctx0, ew.q_w, ew.q_g);
        if (ew.k_w && ew.k_g) k_eff = apply_weight_norm(ctx0, ew.k_w, ew.k_g);
        if (ew.v_w && ew.v_g) v_eff = apply_weight_norm(ctx0, ew.v_w, ew.v_g);
        if (ew.o_w && ew.o_g) o_eff = apply_weight_norm(ctx0, ew.o_w, ew.o_g);

        // Self-attention (POST-norm): h = Norm(h + MHA(h))
        struct ggml_tensor *attn = multi_head_attention(
            ctx0, h, q_eff, ew.q_b, k_eff, ew.k_b,
            v_eff, ew.v_b, o_eff, ew.o_b,
            ew.emb_rel_k, ew.emb_rel_v, n_heads, head_dim, h->ne[1]);
        h = ggml_add(ctx0, h, attn);
        h = layer_norm(ctx0, h, ew.norm1_w, ew.norm1_b, 1e-5f);

        // FFN (POST-norm): h = Norm(h + FFN(h))
        if (ew.norm2_w && ew.conv1_w && ew.conv2_w) {
          struct ggml_tensor *ffn_in = h;
          int k_sz = ew.conv1_w->ne[0] / ew.conv1_w->ne[1];
          if (k_sz <= 0) k_sz = 5;
          struct ggml_tensor *ffn_h = conv1d_im2col(ctx0, h, ew.conv1_w, ew.conv1_b, k_sz, k_sz/2,
                                ew.conv1_w->ne[1], ew.conv1_w->ne[2]);
          ffn_h = ggml_relu(ctx0, ffn_h);
          k_sz = ew.conv2_w->ne[0] / ew.conv2_w->ne[1];
          if (k_sz <= 0) k_sz = 5;
          struct ggml_tensor *ffn_out = conv1d_im2col(ctx0, ffn_h, ew.conv2_w, ew.conv2_b, k_sz, k_sz/2,
                                  ew.conv2_w->ne[1], ew.conv2_w->ne[2]);
          h = ggml_add(ctx0, ffn_out, ffn_in);
          h = layer_norm(ctx0, h, ew.norm2_w, ew.norm2_b, 1e-5f);
        }
      }

      // Post conv: [C_flow, T] → [half, T]
      if (post_w->ne[1] >= C_flow) {
        h = conv1d_im2col(ctx0, h, post_w, post_b, 1, 0, C_flow, half);
      } else {
        h = ggml_mul_mat(ctx0, post_w, h);
        if (post_b) h = ggml_add(ctx0, h, post_b);
      }

      // Reverse coupling (inference: prior→mel): x_b' = x_b - h
      // In VITS/MeloTTS, the flow is called with reverse=True during inference.
      // Forward (training): x_b' = x_b + h (complex→simple)
      // Reverse (inference): x_b' = x_b - h (simple→complex, generative)
      x_b = ggml_sub(ctx0, x_b, h);

    } else {
      // English model: WaveNet-style coupling (original code)
      struct ggml_tensor *h = ggml_mul_mat(ctx0, in_l0_w, x_a);
      int h_half_w = h->ne[0] / 2;
      struct ggml_tensor *h1 = ggml_view_2d(ctx0, h, h_half_w, h->ne[1], h->nb[1], 0);
      struct ggml_tensor *h2 = ggml_view_2d(
          ctx0, h, h->ne[0] - h_half_w, h->ne[1], h->nb[1],
          h_half_w * sizeof(float));
      h = ggml_mul(ctx0, ggml_tanh(ctx0, h1), ggml_sigmoid(ctx0, h2));
      if (in_l1_w) h = ggml_mul_mat(ctx0, in_l1_w, h);
      h = ggml_mul_mat(ctx0, in_l2_w, h);

      int out_half_w = h->ne[0] / 2;
      struct ggml_tensor *mean_w = ggml_view_2d(ctx0, h, out_half_w, h->ne[1], h->nb[1], 0);
      struct ggml_tensor *log_scale_w = ggml_view_2d(
          ctx0, h, h->ne[0] - out_half_w, h->ne[1], h->nb[1],
          out_half_w * sizeof(float));

      x_b = ggml_sub(ctx0, x_b, mean_w);
      x_b = ggml_mul(ctx0, x_b, ggml_exp(ctx0, ggml_neg(ctx0, log_scale_w)));
    }

    // Concat x_a and transformed x_b
    cur = ggml_concat(ctx0, x_a, x_b, 0);
  }

  // Project to n_mels if output channels don't match
  if (cur->ne[0] != n_mels) {
    // Find projection layer
    struct ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    for (auto& [name, t] : w) {
      if (name.find("proj.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) proj_w = t; else proj_b = t;
      } else if (name.find("out_conv.") != std::string::npos) {
        if (name.find("weight") != std::string::npos) proj_w = t; else proj_b = t;
      }
    }

    if (proj_w) {
      cur = ggml_mul_mat(ctx0, proj_w, cur);
      if (proj_b) cur = ggml_add(ctx0, cur, proj_b);
    }
    // else: just leave as-is
  }

  ggml_set_name(cur, "mel_spectrogram");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);

  // --- Execute ---
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: FlowDecoder graph allocation failed");
    ggml_free(ctx0);
    return false;
  }


  // Upload input data (sampled prior z_p)
  ggml_backend_tensor_set(z_p_tensor, state.z_p_expanded.data(), 0,
                          state.z_p_expanded.size() * sizeof(float));

  // Flush relative-position index/mask tensors registered by multi_head_attention
  flush_pending_inputs();

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read output mel spectrogram ---
  int out_ch = cur->ne[0];
  int out_t = cur->ne[1];
  state.mel_spectrogram.resize(out_ch * out_t);
  ggml_backend_tensor_get(cur, state.mel_spectrogram.data(), 0,
                          state.mel_spectrogram.size() * sizeof(float));
  state.total_mel_frames = out_t;

  // Debug: print mel spectrogram value range
  {
    auto& mel = state.mel_spectrogram;
    float mel_min = 1e9, mel_max = -1e9;
    double mel_sum = 0;
    for (size_t i = 0; i < mel.size(); i++) {
      if (mel[i] < mel_min) mel_min = mel[i];
      if (mel[i] > mel_max) mel_max = mel[i];
      mel_sum += mel[i];
    }
    RS_LOG_INFO("OpenVoice2: FlowDecoder mel values [%.4f..%.4f] mean=%.4f",
                mel_min, mel_max, (float)(mel_sum / mel.size()));
  }

  // Subtract per-channel means to remove DC biases (matching VITS inference behavior)
  // The flow decoder can produce large per-channel offsets that the vocoder
  // incorrectly handles, leading to near-silent audio.
  {
    auto& mel = state.mel_spectrogram;
    int C = out_ch;
    int T = out_t;
    for (int c = 0; c < C; c++) {
      double sum = 0;
      for (int t = 0; t < T; t++) {
        sum += mel[c + t * C];
      }
      float mean = (float)(sum / T);
      for (int t = 0; t < T; t++) {
        mel[c + t * C] -= mean;
      }
    }
    float mel_min2 = 1e9, mel_max2 = -1e9;
    double mel_sum2 = 0;
    for (size_t i = 0; i < mel.size(); i++) {
      if (mel[i] < mel_min2) mel_min2 = mel[i];
      if (mel[i] > mel_max2) mel_max2 = mel[i];
      mel_sum2 += mel[i];
    }
    RS_LOG_INFO("OpenVoice2: FlowDecoder after de-mean: [%.4f..%.4f] mean=%.4f",
                mel_min2, mel_max2, (float)(mel_sum2 / mel.size()));
  }

  ggml_free(ctx0);
  RS_LOG_INFO("OpenVoice2: FlowDecoder -> mel [%d, %d]", out_ch, out_t);
  return true;
}

// =====================================================================
// Sub-graph: HiFi-GAN Vocoder
//
// Architecture:
//   1. Conv1D pre-conv (mel → hidden)
//   2. N upsampling blocks: LeakyReLU → TransposedConv → MRF(resblocks)
//   3. Conv1D post-conv (hidden → 1) → tanh
// =====================================================================

bool OpenVoice2Model::RunVocoder(OpenVoice2State& state,
                                  ggml_backend_sched_t sched,
                                  int mel_start, int mel_len) {
  auto& w = weights_.vocoder;
  int n_mels = hparams_.n_mels;
  int T_mel = state.total_mel_frames;
  int hop = hparams_.hop_length;

  if (w.empty()) {
    RS_LOG_WARN("OpenVoice2: no vocoder weights, generating placeholder audio");
    // Fallback: simple oscillator from mel values
    int n_samples = mel_len * hop;
    size_t prev = state.audio_output.size();
    state.audio_output.resize(prev + n_samples, 0.0f);
    for (int i = 0; i < n_samples; i++) {
      int mf = mel_start + i / hop;
      if (mf >= T_mel) break;
      float val = 0.0f;
      for (int m = 0; m < std::min(8, n_mels); m++) {
        val += state.mel_spectrogram[m + mf * n_mels];
      }
      state.audio_output[prev + i] = val * sinf(2.0f * 3.14159f * 440.0f *
          static_cast<float>(i % hop) / hop) * 0.01f;
    }
    return true;
  }

  struct ggml_context *ctx0 = nullptr;
  struct ggml_cgraph *gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, OV2_MAX_NODES)) {
    RS_LOG_ERR("OpenVoice2: failed to create ggml context for Vocoder");
    return false;
  }

  // Input: mel chunk [n_mels, mel_len]
  struct ggml_tensor *mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_mels, mel_len);
  ggml_set_name(mel, "mel_chunk");
  ggml_set_input(mel);

  // Copy mel data
  std::vector<float> mel_data(n_mels * mel_len, 0.0f);
  for (int i = 0; i < mel_len; i++) {
    int frame = mel_start + i;
    if (frame < T_mel) {
      for (int m = 0; m < n_mels; m++) {
        mel_data[m + i * n_mels] = state.mel_spectrogram[m + frame * n_mels];
      }
    }
  }
  // Mel data will be uploaded after graph allocation
  struct ggml_tensor *cur = mel;

  // Find weights
  struct ggml_tensor *pre_w = nullptr, *pre_b = nullptr;
  struct ggml_tensor *post_w = nullptr, *post_b = nullptr;

  // Upsampler weights: up_N.weight, up_N.bias
  std::vector<std::pair<struct ggml_tensor*, struct ggml_tensor*>> up_weights;

  for (auto& [name, t] : w) {
    if (name.find("conv_pre.") != std::string::npos) {
      if (name.find("weight_v") != std::string::npos || name.find("weight") != std::string::npos)
        pre_w = t; else pre_b = t;
    } else if (name.find("conv_post.") != std::string::npos) {
      if (name.find("weight_v") != std::string::npos || name.find("weight") != std::string::npos)
        post_w = t; else post_b = t;
    }
  }

  // Pre-conv: mel → hidden
  if (pre_w) {
    int k_pre = static_cast<int>(pre_w->ne[0]);  // kernel size
    int in_pre = static_cast<int>(pre_w->ne[1]);
    int out_pre = static_cast<int>(pre_w->ne[2]);
    cur = conv1d_im2col(ctx0, cur, pre_w, pre_b, k_pre, k_pre / 2, in_pre, out_pre);
  }

  // --- Upsampling blocks ---
  // Count upsamplers
  int n_ups = 0;
  for (int i = 0; i < 8; i++) {
    std::string pref = "vocoder.ups." + std::to_string(i);
    bool found = false;
    for (auto& [name, t] : w) {
      if (name.find(pref + ".") == 0) { found = true; break; }
    }
    if (found) n_ups++;
    else break;
  }

	  // Build sorted list of ALL residual blocks by global index.
	  // HiFi-GAN resblocks are globally indexed (0, 1, 2, ...) and grouped by
	  // channel count to match each upsampler stage.
	  // Each ResBlock1 contains 3 dilation pairs (convs1/2 with sub-index 0,1,2).
	  struct ResBlockWeights {
	    struct ggml_tensor *c1_w[3] = {}, *c1_g[3] = {}, *c1_b[3] = {};
	    struct ggml_tensor *c2_w[3] = {}, *c2_g[3] = {}, *c2_b[3] = {};
	    int n_pairs = 0;
	  };
	  std::vector<ResBlockWeights> sorted_resblocks;

	  for (int res = 0; res < 30; res++) {
	    ResBlockWeights rb;
	    std::string res_prefix = "vocoder.resblocks." + std::to_string(res) + ".";

	    bool has_any = false;
	    for (auto& [name, t] : w) {
	      if (name.find(res_prefix) != 0) continue;
	      has_any = true;

	      int sub_idx = 0;
	      size_t conv_pos = name.find(".convs1.");
	      if (conv_pos == std::string::npos) conv_pos = name.find(".convs2.");
	      if (conv_pos != std::string::npos) {
	        size_t dot = conv_pos + 7;
	        while (name[dot] >= '0' && name[dot] <= '9') {
	          sub_idx = sub_idx * 10 + (name[dot] - '0');
	          dot++;
	        }
	      }
	      if (sub_idx >= 3) continue;

	      bool is_wv = (name.find("weight_v") != std::string::npos);
	      bool is_wg = (name.find("weight_g") != std::string::npos);
	      bool is_plain_w = (!is_wv && !is_wg && name.find("weight") != std::string::npos);
	      bool is_bias = (name.find("bias") != std::string::npos);

	      if (name.find(".convs1.") != std::string::npos) {
	        if (is_wv || is_plain_w) rb.c1_w[sub_idx] = t;
	        else if (is_wg) rb.c1_g[sub_idx] = t;
	        else if (is_bias) rb.c1_b[sub_idx] = t;
	        if (sub_idx + 1 > rb.n_pairs) rb.n_pairs = sub_idx + 1;
	      } else if (name.find(".convs2.") != std::string::npos) {
	        if (is_wv || is_plain_w) rb.c2_w[sub_idx] = t;
	        else if (is_wg) rb.c2_g[sub_idx] = t;
	        else if (is_bias) rb.c2_b[sub_idx] = t;
	        if (sub_idx + 1 > rb.n_pairs) rb.n_pairs = sub_idx + 1;
	      }
	    }

	    if (!has_any) break;
	    sorted_resblocks.push_back(rb);
	  }


  RS_LOG_INFO("OpenVoice2: loaded %zu resblocks", sorted_resblocks.size());

  // Compute per-stage upsample rates whose product = hop_length
  std::vector<int> upsample_rates(n_ups, 2);
  {
    int remaining = hop;
    for (int i = 0; i < n_ups; i++) {
      if (i == n_ups - 1) {
        upsample_rates[i] = remaining;
      } else {
        int rate = 8;
        int min_remain = 1;
        for (int j = 0; j < (n_ups - i - 1); j++) min_remain *= 2;
        while (rate > 2) {
          if (remaining % rate == 0 && (remaining / rate) >= min_remain) break;
          rate /= 2;
        }
        upsample_rates[i] = rate;
        remaining /= rate;
      }
    }
  }

  size_t res_idx = 0;  // pointer into sorted_resblocks
  for (int up = 0; up < n_ups; up++) {
    int upsample_rate = upsample_rates[up];

	    // LeakyReLU before upsampling (HiFi-GAN: LRELU_SLOPE = 0.1)
	    cur = ggml_leaky_relu(ctx0, cur, 0.1f, false);

    // Find upsampler weight (Chinese: weight_v + weight_g, English: plain weight)
    struct ggml_tensor *up_w = nullptr, *up_g = nullptr, *up_b = nullptr;
    std::string up_pref = "vocoder.ups." + std::to_string(up);
    for (auto& [name, t] : w) {
      if (name.find(up_pref + ".weight_v") != std::string::npos) up_w = t;
      else if (name.find(up_pref + ".weight_g") != std::string::npos) up_g = t;
      else if (name.find(up_pref + ".weight") != std::string::npos) up_w = t;
      else if (name.find(up_pref + ".bias") != std::string::npos) up_b = t;
    }

    if (up_w) {
      // Apply weight normalization if weight_g is present
      struct ggml_tensor *eff_w = up_w;
      if (up_g) {
        eff_w = apply_weight_norm(ctx0, up_w, up_g);
      }
      // ggml_conv_transpose_1d expects data [OW, IC, N] — transpose cur
      struct ggml_tensor *cur_t = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

      struct ggml_tensor *up_out = ggml_conv_transpose_1d(
          ctx0, eff_w, cur_t,
          upsample_rate,  // stride
          0,               // padding (must be 0 for this ggml version)
          1);              // dilation

      // Result: [OL, OC, N] — transpose back to [OC, OL]
      up_out = ggml_cont(ctx0, ggml_transpose(ctx0, up_out));
      int out_ch = static_cast<int>(up_out->ne[0]);
      int out_len = static_cast<int>(up_out->ne[1]);
      up_out = ggml_reshape_2d(ctx0, up_out, out_ch, out_len);
      // Add bias after transposed conv
      if (up_b) {
        struct ggml_tensor *bias_2d = ggml_reshape_2d(ctx0, up_b, out_ch, 1);
        up_out = ggml_add(ctx0, up_out, bias_2d);
      }

      cur = up_out;
    } else {
      RS_LOG_WARN("OpenVoice2: upsampler %d not found, skipping", up);
      continue;
    }

    // MRF residual blocks: apply blocks with matching channel count.
    // Each ResBlock1 has 3 dilation pairs (dil=[1,3,5] for convs1, dil=1 for convs2).
    int cur_ch = static_cast<int>(cur->ne[0]);
    const int dilations[] = {1, 3, 5};
    while (res_idx < sorted_resblocks.size()) {
      int rb_applied = 0;
      int rb_total_pairs = 0;
      auto& rb = sorted_resblocks[res_idx];
      if (!rb.c1_w[0] || !rb.c2_w[0]) { res_idx++; continue; }
      if (rb_applied > 0) RS_LOG_INFO("OpenVoice2: stage up=%d applied %d resblocks with %d total pairs", up, rb_applied, rb_total_pairs);
      if (rb.c1_w[0]->ne[1] != cur_ch) break;  // different stage group

      struct ggml_tensor *residual = cur;

      for (int p = 0; p < rb.n_pairs; p++) {
        if (!rb.c1_w[p] || !rb.c2_w[p]) continue;
        int in_ch = static_cast<int>(rb.c1_w[p]->ne[1]);
        int out_ch_rb = static_cast<int>(rb.c1_w[p]->ne[2]);
        int k_size = static_cast<int>(rb.c1_w[p]->ne[0]);
        int dil = dilations[p];
        int pad = (k_size * dil - dil) / 2;  // PyTorch get_padding

        // Apply weight norm if weight_g present
        struct ggml_tensor *eff_c1 = rb.c1_w[p];
        struct ggml_tensor *eff_c2 = rb.c2_w[p];
        if (rb.c1_g[p]) eff_c1 = apply_weight_norm(ctx0, rb.c1_w[p], rb.c1_g[p]);
        if (rb.c2_g[p]) eff_c2 = apply_weight_norm(ctx0, rb.c2_w[p], rb.c2_g[p]);

        // First conv (dilated)
        cur = ggml_leaky_relu(ctx0, cur, 0.1f, false);
        cur = conv1d_im2col(ctx0, cur, eff_c1, rb.c1_b[p],
                            k_size, pad, in_ch, out_ch_rb, dil);

        cur = ggml_leaky_relu(ctx0, cur, 0.1f, false);

        // Second conv (dilation=1 always)
        in_ch = static_cast<int>(eff_c2->ne[1]);
        out_ch_rb = static_cast<int>(eff_c2->ne[2]);
        k_size = static_cast<int>(eff_c2->ne[0]);
        cur = conv1d_im2col(ctx0, cur, eff_c2, rb.c2_b[p],
                            k_size, k_size / 2, in_ch, out_ch_rb, 1);
      }

      rb_applied++;
      // Residual connection
      cur = ggml_add(ctx0, cur, residual);
      res_idx++;
    }
  }

  // Post-conv: hidden → 1 (audio)
  struct ggml_tensor *pre_tanh = nullptr;
  if (post_w) {
    cur = ggml_leaky_relu(ctx0, cur, 0.1f, false);  // LeakyReLU before post-conv
    int k_post = static_cast<int>(post_w->ne[0]);
    int in_post = static_cast<int>(post_w->ne[1]);
    cur = conv1d_im2col(ctx0, cur, post_w, post_b, k_post, k_post / 2, in_post, 1);
    pre_tanh = cur;
    cur = ggml_tanh(ctx0, cur);
  }

  ggml_set_name(cur, "audio_output");
  ggml_set_output(cur);
  ggml_build_forward_expand(gf, cur);
  if (pre_tanh) {
    ggml_set_output(pre_tanh);
    ggml_build_forward_expand(gf, pre_tanh);
  }

  // --- Execute ---
  ggml_backend_sched_reset(sched);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("OpenVoice2: Vocoder graph allocation failed");
    ggml_free(ctx0);
    return false;
  }

  // Upload mel data after graph allocation
  ggml_backend_tensor_set(mel, mel_data.data(), 0,
                          mel_data.size() * sizeof(float));

  ggml_backend_sched_graph_compute(sched, gf);

  // --- Read audio output ---
  int n_samples = cur->ne[0] * cur->ne[1];
  if (n_samples <= 0) {
    ggml_free(ctx0);
    return false;
  }

  std::vector<float> audio(n_samples);
  ggml_backend_tensor_get(cur, audio.data(), 0, n_samples * sizeof(float));

  // Debug: print audio value range
  {
    float amin = 1e9, amax = -1e9;
    double asum = 0;
    for (int i = 0; i < n_samples; i++) {
      if (audio[i] < amin) amin = audio[i];
      if (audio[i] > amax) amax = audio[i];
      asum += audio[i];
    }
    RS_LOG_INFO("OpenVoice2: audio chunk values [%.4f..%.4f] mean=%.4f",
                amin, amax, (float)(asum / n_samples));
  }
  // Also read pre-tanh values if available
  if (pre_tanh) {
    int n_pt = pre_tanh->ne[0] * pre_tanh->ne[1];
    std::vector<float> pt_data(n_pt);
    ggml_backend_tensor_get(pre_tanh, pt_data.data(), 0, n_pt * sizeof(float));
    float ptmin = 1e9, ptmax = -1e9;
    double ptsum = 0;
    for (int i = 0; i < n_pt; i++) {
      if (pt_data[i] < ptmin) ptmin = pt_data[i];
      if (pt_data[i] > ptmax) ptmax = pt_data[i];
      ptsum += pt_data[i];
    }
    RS_LOG_INFO("OpenVoice2: pre-tanh values [%.4f..%.4f] mean=%.4f",
                ptmin, ptmax, (float)(ptsum / n_pt));
  }

  // Append to state's audio buffer
  size_t prev_size = state.audio_output.size();
  state.audio_output.resize(prev_size + n_samples);
  std::memcpy(state.audio_output.data() + prev_size, audio.data(),
              n_samples * sizeof(float));

  ggml_free(ctx0);
  int n_audio = n_samples;
  RS_LOG_INFO("OpenVoice2: Vocoder chunk [%d..%d] -> %d samples",
              mel_start, mel_start + mel_len, n_audio);
  return true;
}

// =====================================================================
// Sub-graph: Tone Color Encoder (voice cloning reference encoder)
//
// Architecture: Mel → Conv2D blocks → Global Pooling → Style Embedding
// =====================================================================

bool OpenVoice2Model::RunToneColorEncoder(OpenVoice2State& state,
                                           const std::vector<float>& mel,
                                           ggml_backend_sched_t sched) {
  auto& w = converter_weights_.all_tensors;
  if (w.empty() || !converter_weights_.loaded) {
    RS_LOG_WARN("OpenVoice2: no tone color converter weights loaded");
    state.has_tone_embedding = false;
    return true;
  }

  RS_LOG_INFO("OpenVoice2: ToneColorEncoder not yet implemented (needs converter model)");
  state.has_tone_embedding = false;
  return true;
}

// =====================================================================
// Static registration
// =====================================================================
namespace {
struct OpenVoice2Registrar {
  OpenVoice2Registrar() {
    rs_register_model_arch("openvoice2", []() {
      return std::make_shared<OpenVoice2Model>();
    });
  }
} global_openvoice2_reg;
}  // namespace
