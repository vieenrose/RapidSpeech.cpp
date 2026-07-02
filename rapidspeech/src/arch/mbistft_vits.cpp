#include "mbistft_vits.h"
#include "core/rs_context.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "gguf.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define MBV_MAX_NODES 8192

// =====================================================================
// Pending-inputs registry (no_alloc graph pattern, same as openvoice2):
// input tensors have data=nullptr until ggml_backend_sched_alloc_graph;
// register intended contents here and flush after allocation.
// =====================================================================
namespace {

struct MbvPendingInput {
  struct ggml_tensor* tensor;
  std::vector<uint8_t> data;
};

thread_local std::vector<MbvPendingInput> g_mbv_pending;

void mbv_register_input(struct ggml_tensor* t, const void* data, size_t size) {
  MbvPendingInput pi;
  pi.tensor = t;
  pi.data.resize(size);
  if (size > 0) std::memcpy(pi.data.data(), data, size);
  g_mbv_pending.push_back(std::move(pi));
}

void mbv_flush_inputs() {
  for (auto& pi : g_mbv_pending) {
    ggml_backend_tensor_set(pi.tensor, pi.data.data(), 0, pi.data.size());
  }
  g_mbv_pending.clear();
}

// =====================================================================
// Backend scheduling policy (env-gated; no effect on numerics).
//
//   MBISTFT_PROFILE     : run each stage node-by-node and print per-node
//                         op/shape/backend/ms to stderr (the sched
//                         synchronizes before the callback, so timing is
//                         exact — used to locate the watchdog-tripping op).
//   MBISTFT_ENC_CPU     : pin the whole encoder+DP graph to the CPU backend.
//   MBISTFT_FLOW_CPU    : pin the whole reverse-flow graph to the CPU backend.
//   MBISTFT_DEC_CPU     : pin the whole MB-iSTFT decoder graph to CPU.
//   MBISTFT_DEC_CT_CPU  : pin ONLY the conv_transpose_1d ops of the decoder
//                         to CPU (the quadratic-in-length CUDA kernel that
//                         trips the Nano's 2s display watchdog), leaving the
//                         matmul/im2col ops on the GPU.
// All pinning routes cross-backend copies through the sched automatically, so
// results are bit-for-bit identical to the all-GPU / all-CPU paths.
// =====================================================================

thread_local std::chrono::high_resolution_clock::time_point g_mbv_prof_t0;

bool mbv_prof_cb(struct ggml_tensor* t, bool ask, void* ud) {
  (void)ud;
  if (ask) {
    g_mbv_prof_t0 = std::chrono::high_resolution_clock::now();
    return true;  // observe this node
  }
  double ms = std::chrono::duration<double, std::milli>(
      std::chrono::high_resolution_clock::now() - g_mbv_prof_t0).count();
  ggml_backend_buffer_t buf = t->buffer;
  const char* bname = buf ? ggml_backend_buffer_name(buf) : "?";
  fprintf(stderr,
          "[mbv-prof] %-16s op=%-18s ne=[%5lld,%5lld,%3lld,%2lld] buf=%-10s %8.2f ms\n",
          t->name[0] ? t->name : "(anon)", ggml_op_name(t->op),
          (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
          (long long)t->ne[3], bname, ms);
  return true;
}

ggml_backend_t mbv_cpu_backend(ggml_backend_sched_t sched) {
  int n = ggml_backend_sched_get_n_backends(sched);
  for (int i = 0; i < n; i++) {
    ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
    if (ggml_backend_is_cpu(b)) return b;
  }
  return nullptr;
}

// Must be called AFTER ggml_backend_sched_reset and BEFORE
// ggml_backend_sched_alloc_graph (set_tensor_backend feeds split_graph).
void mbv_apply_sched_policy(ggml_backend_sched_t sched, struct ggml_cgraph* gf,
                            bool all_cpu, bool convt_cpu) {
  ggml_backend_sched_set_eval_callback(
      sched, getenv("MBISTFT_PROFILE") ? mbv_prof_cb : nullptr, nullptr);
  if (!all_cpu && !convt_cpu) return;
  ggml_backend_t cpu = mbv_cpu_backend(sched);
  if (!cpu) return;  // CPU-only sched: everything is already on CPU
  const int n = ggml_graph_n_nodes(gf);
  int pinned = 0;
  for (int i = 0; i < n; i++) {
    struct ggml_tensor* node = ggml_graph_node(gf, i);
    if (all_cpu || (convt_cpu && node->op == GGML_OP_CONV_TRANSPOSE_1D)) {
      ggml_backend_sched_set_tensor_backend(sched, node, cpu);
      pinned++;
    }
  }
  if (getenv("MBISTFT_PROFILE"))
    fprintf(stderr, "[mbv-prof] pinned %d/%d nodes to CPU%s\n", pinned, n,
            convt_cpu && !all_cpu ? " (conv_transpose_1d only)" : "");
}

// Create an F32 input tensor and register its contents.
struct ggml_tensor* mbv_input_f32(struct ggml_context* ctx, const char* name,
                                  const float* data, int64_t ne0, int64_t ne1 = 1,
                                  int64_t ne2 = 1) {
  struct ggml_tensor* t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
  ggml_set_name(t, name);
  ggml_set_input(t);
  mbv_register_input(t, data, (size_t)ne0 * ne1 * ne2 * sizeof(float));
  return t;
}

struct ggml_tensor* mbv_input_i32(struct ggml_context* ctx, const char* name,
                                  const int32_t* data, int64_t n) {
  struct ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
  ggml_set_name(t, name);
  ggml_set_input(t);
  mbv_register_input(t, data, (size_t)n * sizeof(int32_t));
  return t;
}

// =====================================================================
// Graph building helpers.  Two data layouts are used:
//   [C,T]: ne0 = channels (fastest), ne1 = time.  Used by the encoder,
//          duration predictor and flow (LayerNorm/1x1-conv friendly).
//   [T,C]: ne0 = time, ne1 = channels.  Used by the decoder (conv-native).
// All weights come from the GGUF with PyTorch memory order, i.e. a
// Conv1d weight [Cout, Cin, K] appears as ne = (K, Cin, Cout) and a
// ConvTranspose1d weight [Cin, Cout, K] as ne = (K, Cout, Cin) — both are
// exactly what ggml_im2col / ggml_conv_transpose_1d expect.
// =====================================================================

// 1x1 conv (= linear) on [C,T] data: w ne (1, Cin, Cout) or (Cin, Cout).
struct ggml_tensor* mbv_linear_ct(struct ggml_context* ctx, struct ggml_tensor* x,
                                  struct ggml_tensor* w, struct ggml_tensor* b) {
  struct ggml_tensor* w2d = w;
  if (w->ne[0] == 1)  // 1x1 conv weight (1, Cin, Cout) -> (Cin, Cout)
    w2d = ggml_reshape_2d(ctx, w, w->ne[1], w->ne[2]);
  struct ggml_tensor* out = ggml_mul_mat(ctx, w2d, x);  // (Cout, T)
  if (b) out = ggml_add(ctx, out, b);
  return out;
}

// Conv1d on [T,C] data (all-F32 im2col path; ggml_conv_1d would round the
// im2col buffer to F16, hurting parity).  Returns [OL, OC].
struct ggml_tensor* mbv_conv1d_tc(struct ggml_context* ctx, struct ggml_tensor* x,
                                  struct ggml_tensor* w, struct ggml_tensor* b,
                                  int pad, int dil) {
  const int64_t T   = x->ne[0];
  const int64_t Cin = x->ne[1];
  const int64_t K   = w->ne[0];
  const int64_t OC  = w->ne[2];
  struct ggml_tensor* x3 = ggml_reshape_3d(ctx, x, T, Cin, 1);
  struct ggml_tensor* im = ggml_im2col(ctx, w, x3, 1, 0, pad, 0, dil, 0,
                                       /*is_2D=*/false, GGML_TYPE_F32);  // (Cin*K, OL, 1)
  struct ggml_tensor* out = ggml_mul_mat(
      ctx, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]),
      ggml_reshape_2d(ctx, w, K * Cin, OC));  // (OL, OC)
  if (b) out = ggml_add(ctx, out, ggml_reshape_2d(ctx, b, 1, OC));
  return out;
}

// Conv1d on [C,T] data.  Returns [OC, OL].
struct ggml_tensor* mbv_conv1d_ct(struct ggml_context* ctx, struct ggml_tensor* x,
                                  struct ggml_tensor* w, struct ggml_tensor* b,
                                  int pad, int dil) {
  struct ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));  // (T, Cin)
  const int64_t T   = xt->ne[0];
  const int64_t Cin = xt->ne[1];
  const int64_t K   = w->ne[0];
  const int64_t OC  = w->ne[2];
  struct ggml_tensor* x3 = ggml_reshape_3d(ctx, xt, T, Cin, 1);
  struct ggml_tensor* im = ggml_im2col(ctx, w, x3, 1, 0, pad, 0, dil, 0,
                                       false, GGML_TYPE_F32);
  struct ggml_tensor* mm = ggml_mul_mat(
      ctx, ggml_reshape_2d(ctx, im, im->ne[0], im->ne[1] * im->ne[2]),
      ggml_reshape_2d(ctx, w, K * Cin, OC));  // (OL, OC)
  struct ggml_tensor* out = ggml_cont(ctx, ggml_transpose(ctx, mm));  // (OC, OL)
  if (b) out = ggml_add(ctx, out, b);
  return out;
}

// Drop-in replacement for ggml_conv_transpose_1d(w, x, s, 0, 1) that stays on
// the GPU.  ggml's CUDA conv_transpose_1d kernel is O(out*IC*L) with a
// per-output stride scan over the full input length — on the decoder's long
// axes (upsampling, iSTFT-OLA, PQMF) a single launch runs >2s and trips the
// Jetson Nano's 2s display watchdog.  We instead zero-insert the input (insert
// s-1 zeros after each sample) and run a regular F32 conv1d with the
// flipped/transposed weight: mathematically identical to a stride-s transpose
// conv, but it uses the fast im2col + cuBLAS path (no watchdog, no CPU offload).
//   w: PyTorch ConvTranspose1d weight, ne=(K, OC, IC).
//   x: input [L, IC]  (ne0 = length).
// Returns (OL_full, OC, 1), OL_full = (L-1)*s + K — identical shape to
// ggml_conv_transpose_1d's unpadded output.
struct ggml_tensor* mbv_convt1d_reform(struct ggml_context* ctx,
                                       struct ggml_tensor* w,
                                       struct ggml_tensor* x, int s) {
  const int64_t K  = w->ne[0];
  const int64_t OC = w->ne[1];
  const int64_t IC = w->ne[2];
  const int64_t L  = x->ne[0];
  const int64_t OL_full = (L - 1) * s + K;
  static int ct_uid = 0;

  // zero-insert along the time axis:
  //   (L,IC) -> (1,L,IC) -> pad ne0 to s -> (s,L,IC) -> (s*L, IC)
  // u[l*s]=x[l], u[l*s+j]=0 for 0<j<s (ggml_pad fills with zeros).
  struct ggml_tensor* xr = ggml_reshape_3d(ctx, ggml_cont(ctx, x), 1, L, IC);
  struct ggml_tensor* xp = ggml_pad(ctx, xr, (int)s - 1, 0, 0, 0);  // (s,L,IC)
  struct ggml_tensor* u  = ggml_reshape_2d(ctx, xp, s * L, IC);     // (s*L,IC)

  // conv weight: permute (K,OC,IC)->(K,IC,OC) then flip along K (ne0).
  struct ggml_tensor* wp = ggml_cont(ctx, ggml_permute(ctx, w, 0, 2, 1, 3));
  std::vector<int32_t> rev((size_t)K);
  for (int i = 0; i < (int)K; i++) rev[i] = (int32_t)(K - 1 - i);
  char nm[48];
  snprintf(nm, sizeof(nm), "ct_revk_%d", ct_uid++);
  struct ggml_tensor* rk = mbv_input_i32(ctx, nm, rev.data(), K);
  struct ggml_tensor* wf = ggml_reshape_2d(ctx, wp, K, IC * OC);   // (K, IC*OC)
  wf = ggml_cont(ctx, ggml_transpose(ctx, wf));                    // (IC*OC, K)
  wf = ggml_get_rows(ctx, wf, rk);                                 // flip along K
  wf = ggml_cont(ctx, ggml_transpose(ctx, wf));                    // (K, IC*OC)
  struct ggml_tensor* wconv = ggml_reshape_3d(ctx, wf, K, IC, OC); // (K, IC, OC)

  // regular F32 conv1d with pad = K-1, dilation 1  ->  (s*L+K-1, OC)
  struct ggml_tensor* y = mbv_conv1d_tc(ctx, u, wconv, nullptr, (int)K - 1, 1);
  // the transpose-conv output is the leading OL_full rows (the s-1 tail rows
  // are structurally zero — no valid input taps reach them).
  struct ggml_tensor* yt = ggml_cont(ctx,
      ggml_view_2d(ctx, y, OL_full, OC, y->nb[1], 0));
  return ggml_reshape_3d(ctx, yt, OL_full, OC, 1);
}

// Dispatch: default to the GPU-friendly zero-insert+conv1d reformulation (it is
// numerically identical, uses only portable ops, and dodges ggml's watchdog-
// tripping CUDA conv_transpose_1d kernel).  Escape hatch MBISTFT_CT_STOCK=1
// forces the stock ggml_conv_transpose_1d (e.g. for A/B or non-CUDA backends).
struct ggml_tensor* mbv_convt1d(struct ggml_context* ctx, struct ggml_tensor* w,
                                struct ggml_tensor* x, int s) {
  static const bool stock = getenv("MBISTFT_CT_STOCK") != nullptr;
  if (stock) return ggml_conv_transpose_1d(ctx, w, x, s, 0, 1);
  return mbv_convt1d_reform(ctx, w, x, s);
}

// LayerNorm over the channel dim of [C,T] data (modules.LayerNorm, eps=1e-5).
struct ggml_tensor* mbv_layer_norm_ct(struct ggml_context* ctx, struct ggml_tensor* x,
                                      struct ggml_tensor* gamma, struct ggml_tensor* beta) {
  x = ggml_norm(ctx, x, 1e-5f);
  x = ggml_mul(ctx, x, gamma);
  x = ggml_add(ctx, x, beta);
  return x;
}

// Windowed relative-position multi-head self-attention (Shaw et al.),
// matching attentions.MultiHeadAttention with window_size=4, heads_share=true.
// x: [192,T].  emb_rel_k/v: ne (head_dim, 2*ws+1, 1).  Returns conv_o output
// [192,T] (the MHA sublayer output BEFORE residual/LayerNorm).
struct ggml_tensor* mbv_rel_mha(struct ggml_context* ctx, struct ggml_tensor* x,
                                struct ggml_tensor* q_w, struct ggml_tensor* q_b,
                                struct ggml_tensor* k_w, struct ggml_tensor* k_b,
                                struct ggml_tensor* v_w, struct ggml_tensor* v_b,
                                struct ggml_tensor* o_w, struct ggml_tensor* o_b,
                                struct ggml_tensor* emb_rel_k,
                                struct ggml_tensor* emb_rel_v,
                                int n_heads, int head_dim, int T, int window_size,
                                int layer_idx) {
  const int hidden = n_heads * head_dim;
  const int ws     = 2 * window_size + 1;  // 9
  const float scale = 1.0f / sqrtf((float)head_dim);

  struct ggml_tensor* Q = mbv_linear_ct(ctx, x, q_w, q_b);  // (192, T)
  struct ggml_tensor* K = mbv_linear_ct(ctx, x, k_w, k_b);
  struct ggml_tensor* V = mbv_linear_ct(ctx, x, v_w, v_b);

  // (hd, nh, T) -> (hd, T, nh): channel dim d = h*head_dim + dk (dk fastest),
  // matching PyTorch's view(b, n_heads, k_channels, t).
  Q = ggml_cont(ctx, ggml_permute(ctx,
      ggml_reshape_4d(ctx, Q, head_dim, n_heads, T, 1), 0, 2, 1, 3));
  K = ggml_cont(ctx, ggml_permute(ctx,
      ggml_reshape_4d(ctx, K, head_dim, n_heads, T, 1), 0, 2, 1, 3));
  V = ggml_cont(ctx, ggml_permute(ctx,
      ggml_reshape_4d(ctx, V, head_dim, n_heads, T, 1), 0, 2, 1, 3));

  // Content scores: (T_q, T_k, nh) — ne0 = query index.
  struct ggml_tensor* scores = ggml_mul_mat(ctx, Q, K);

  // --- rel-pos K bias:  bias[q,k,h] = dot(Q[:,q,h], emb_rel_k[:, (k-q)+ws0])
  // for |k-q| <= window_size, else 0 (zero-padded embedding table).
  {
    struct ggml_tensor* erk = ggml_reshape_2d(ctx, emb_rel_k, head_dim, ws);
    struct ggml_tensor* Q2d = ggml_reshape_2d(ctx, Q, head_dim, (int64_t)T * n_heads);
    struct ggml_tensor* qdot = ggml_mul_mat(ctx, Q2d, erk);  // (T*nh, ws)
    // qdot flat index of (t, h, r) = t + T*h + T*nh*r
    const int64_t total = (int64_t)T * T * n_heads;
    std::vector<int32_t> idx(total);
    std::vector<float>   mask(total, 0.0f);
    for (int h = 0; h < n_heads; h++) {
      for (int k = 0; k < T; k++) {
        for (int q = 0; q < T; q++) {
          const int r = k - q;
          const bool in = (r >= -window_size && r <= window_size);
          const int r_idx = in ? (r + window_size) : 0;
          const int64_t pos = (int64_t)q + (int64_t)T * k + (int64_t)T * T * h;
          idx[pos] = (int32_t)((int64_t)q + (int64_t)T * h +
                               (int64_t)T * n_heads * r_idx);
          mask[pos] = in ? 1.0f : 0.0f;
        }
      }
    }
    char nb[64];
    snprintf(nb, sizeof(nb), "relk_idx_l%d", layer_idx);
    struct ggml_tensor* idx_t = mbv_input_i32(ctx, nb, idx.data(), total);
    snprintf(nb, sizeof(nb), "relk_mask_l%d", layer_idx);
    struct ggml_tensor* mask_t = mbv_input_f32(ctx, nb, mask.data(), total);

    struct ggml_tensor* rows = ggml_reshape_2d(
        ctx, qdot, 1, (int64_t)T * n_heads * ws);
    struct ggml_tensor* bias = ggml_get_rows(ctx, rows, idx_t);  // (1, total)
    bias = ggml_mul(ctx, ggml_reshape_1d(ctx, bias, total),
                    ggml_reshape_1d(ctx, mask_t, total));
    bias = ggml_reshape_4d(ctx, bias, T, T, n_heads, 1);
    scores = ggml_add(ctx, scores, bias);
  }

  // softmax over the key dim: transpose to (T_k, T_q, nh)
  struct ggml_tensor* scores_t = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
  struct ggml_tensor* p = ggml_soft_max_ext(ctx, scores_t, nullptr, scale, 0.0f);

  // out = p @ V : (hd, T_q, nh)
  struct ggml_tensor* out = ggml_mul_mat(
      ctx, ggml_cont(ctx, ggml_transpose(ctx, V)), p);

  // --- rel-pos V correction: out[:,q,h] += sum_r p[q, q+r, h] * emb_rel_v[:, r]
  {
    struct ggml_tensor* erv = ggml_reshape_2d(ctx, emb_rel_v, head_dim, ws);
    struct ggml_tensor* erv_t = ggml_cont(ctx, ggml_transpose(ctx, erv));  // (ws, hd)
    // gather attn_by_rel (ws, T, nh):  value = p[k=q+r-ws0, q, h]
    // p flat index of (k, q, h) = k + T*q + T*T*h
    const int64_t total = (int64_t)ws * T * n_heads;
    std::vector<int32_t> idx(total);
    std::vector<float>   mask(total, 0.0f);
    for (int h = 0; h < n_heads; h++) {
      for (int q = 0; q < T; q++) {
        for (int r_idx = 0; r_idx < ws; r_idx++) {
          const int k = q + r_idx - window_size;
          const int64_t pos = (int64_t)r_idx + (int64_t)ws * q + (int64_t)ws * T * h;
          if (k >= 0 && k < T) {
            idx[pos]  = (int32_t)((int64_t)k + (int64_t)T * q + (int64_t)T * T * h);
            mask[pos] = 1.0f;
          } else {
            idx[pos]  = 0;
            mask[pos] = 0.0f;
          }
        }
      }
    }
    char nb[64];
    snprintf(nb, sizeof(nb), "relv_idx_l%d", layer_idx);
    struct ggml_tensor* idx_t = mbv_input_i32(ctx, nb, idx.data(), total);
    snprintf(nb, sizeof(nb), "relv_mask_l%d", layer_idx);
    struct ggml_tensor* mask_t = mbv_input_f32(ctx, nb, mask.data(), total);

    struct ggml_tensor* prow = ggml_reshape_2d(ctx, ggml_cont(ctx, p), 1,
                                               (int64_t)T * T * n_heads);
    struct ggml_tensor* g = ggml_get_rows(ctx, prow, idx_t);  // (1, total)
    g = ggml_mul(ctx, ggml_reshape_1d(ctx, g, total),
                 ggml_reshape_1d(ctx, mask_t, total));
    struct ggml_tensor* abr = ggml_reshape_3d(ctx, g, ws, T, n_heads);
    struct ggml_tensor* corr = ggml_mul_mat(ctx, erv_t, abr);  // (hd, T, nh)
    out = ggml_add(ctx, out, corr);
  }

  // merge heads: (hd, T, nh) -> (hd, nh, T) -> (192, T)
  struct ggml_tensor* merged = ggml_reshape_2d(
      ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)), hidden, T);

  struct ggml_tensor* y = mbv_linear_ct(ctx, merged, o_w, o_b);
  return y;
}

// Channel flip (torch.flip dim=1) of [C,T] data via get_rows on the transposed
// view; rev_idx is a shared I32 input holding {C-1, ..., 0}.
struct ggml_tensor* mbv_flip_channels(struct ggml_context* ctx, struct ggml_tensor* x,
                                      struct ggml_tensor* rev_idx) {
  struct ggml_tensor* xt = ggml_cont(ctx, ggml_transpose(ctx, x));  // (T, C)
  struct ggml_tensor* fl = ggml_get_rows(ctx, xt, rev_idx);         // (T, C)
  return ggml_cont(ctx, ggml_transpose(ctx, fl));                   // (C, T)
}

}  // namespace

// =====================================================================
// MBIstftVitsModel
// =====================================================================

MBIstftVitsModel::MBIstftVitsModel() {}
MBIstftVitsModel::~MBIstftVitsModel() = default;

struct ggml_tensor* MBIstftVitsModel::Get(const std::string& name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end()) {
    RS_LOG_ERR("mbistft-vits: missing tensor '%s'", name.c_str());
    return nullptr;
  }
  return it->second;
}

bool MBIstftVitsModel::Load(const std::unique_ptr<rs_context_t>& ctx,
                            ggml_backend_t backend) {
  (void)backend;
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("mbistft-vits: invalid context");
    return false;
  }
  gguf_context* g = ctx->ctx_gguf;

  auto read_i32 = [&](const char* key, int32_t& dst) {
    int64_t k = gguf_find_key(g, key);
    if (k == -1) return;
    switch (gguf_get_kv_type(g, k)) {
      case GGUF_TYPE_INT32:  dst = gguf_get_val_i32(g, k); break;
      case GGUF_TYPE_UINT32: dst = (int32_t)gguf_get_val_u32(g, k); break;
      case GGUF_TYPE_INT64:  dst = (int32_t)gguf_get_val_i64(g, k); break;
      case GGUF_TYPE_UINT64: dst = (int32_t)gguf_get_val_u64(g, k); break;
      case GGUF_TYPE_INT16:  dst = gguf_get_val_i16(g, k); break;
      case GGUF_TYPE_UINT16: dst = gguf_get_val_u16(g, k); break;
      case GGUF_TYPE_INT8:   dst = gguf_get_val_i8(g, k); break;
      case GGUF_TYPE_UINT8:  dst = gguf_get_val_u8(g, k); break;
      default:
        RS_LOG_WARN("mbistft-vits: KV '%s' has non-integer type %d", key,
                    (int)gguf_get_kv_type(g, k));
    }
  };
  auto read_arr = [&](const char* key, std::vector<int32_t>& dst) {
    int64_t k = gguf_find_key(g, key);
    if (k == -1) return;
    const int n = (int)gguf_get_arr_n(g, k);
    const enum gguf_type at = gguf_get_arr_type(g, k);
    dst.resize(n);
    if (at == GGUF_TYPE_INT32 || at == GGUF_TYPE_UINT32) {
      const int32_t* d = static_cast<const int32_t*>(gguf_get_arr_data(g, k));
      std::copy(d, d + n, dst.begin());
    } else if (at == GGUF_TYPE_INT64 || at == GGUF_TYPE_UINT64) {
      const int64_t* d = static_cast<const int64_t*>(gguf_get_arr_data(g, k));
      for (int i = 0; i < n; i++) dst[i] = (int32_t)d[i];
    } else {
      RS_LOG_WARN("mbistft-vits: KV array '%s' has unsupported type %d", key, (int)at);
      dst.clear();
    }
  };
  read_i32("mbistft.n_vocab",                  hparams_.n_vocab);
  read_i32("mbistft.num_tones",                hparams_.num_tones);
  read_i32("mbistft.num_langs",                hparams_.num_langs);
  read_i32("mbistft.hidden_channels",          hparams_.hidden_channels);
  read_i32("mbistft.inter_channels",           hparams_.inter_channels);
  read_i32("mbistft.filter_channels",          hparams_.filter_channels);
  read_i32("mbistft.n_heads",                  hparams_.n_heads);
  read_i32("mbistft.n_layers",                 hparams_.n_layers);
  read_i32("mbistft.kernel_size",              hparams_.kernel_size);
  read_i32("mbistft.window_size",              hparams_.window_size);
  read_i32("mbistft.k_channels",               hparams_.k_channels);
  read_i32("mbistft.subbands",                 hparams_.subbands);
  read_i32("mbistft.gen_istft_n_fft",          hparams_.gen_istft_n_fft);
  read_i32("mbistft.gen_istft_hop_size",       hparams_.gen_istft_hop);
  read_i32("mbistft.upsample_initial_channel", hparams_.ups_initial_ch);
  read_i32("mbistft.sampling_rate",            hparams_.sample_rate);
  read_i32("mbistft.hop_length",               hparams_.hop_length);
  read_arr("mbistft.upsample_rates",           hparams_.upsample_rates);
  read_arr("mbistft.upsample_kernel_sizes",    hparams_.upsample_kernel_sizes);
  read_arr("mbistft.resblock_kernel_sizes",    hparams_.resblock_kernel_sizes);

  meta_.arch_name = "mbistft-vits";
  meta_.audio_sample_rate = hparams_.sample_rate;
  meta_.vocab_size = hparams_.n_vocab;

  const int n_tensors = (int)gguf_get_n_tensors(g);
  for (int i = 0; i < n_tensors; i++) {
    const char* name = gguf_get_tensor_name(g, i);
    struct ggml_tensor* t = ggml_get_tensor(ctx->gguf_data, name);
    if (t) tensors_[name] = t;
  }
  RS_LOG_INFO("mbistft-vits: %zu tensors, hidden=%d layers=%d sr=%d",
              tensors_.size(), hparams_.hidden_channels, hparams_.n_layers,
              hparams_.sample_rate);

  // dp filter channels from the conv_1 weight shape
  if (auto* t = Get("dp.conv_1.weight")) hparams_.dp_filter_channels = (int)t->ne[2];

  // host copy of the iSTFT analysis window (for the OLA envelope kernel)
  if (auto* t = Get("istft.window")) {
    istft_window_.resize(t->ne[0]);
    ggml_backend_tensor_get(t, istft_window_.data(), 0,
                            istft_window_.size() * sizeof(float));
  } else {
    return false;
  }
  return true;
}

std::shared_ptr<RSState> MBIstftVitsModel::CreateState() {
  return std::make_shared<MBIstftVitsState>();
}

bool MBIstftVitsModel::PushText(RSState& state, const char* text,
                                const char* language, const char* instruct) {
  (void)language; (void)instruct;
  auto& s = static_cast<MBIstftVitsState&>(state);
  if (!s.phone_ids.empty() &&
      s.phone_ids.size() == s.tone_ids.size() &&
      s.phone_ids.size() == s.lang_ids.size()) {
    return true;  // precomputed ids already pushed into the state
  }
  RS_LOG_ERR("mbistft-vits: no text frontend yet — push precomputed "
             "phone/tone/lang ids into the state (text was '%s')",
             text ? text : "");
  return false;
}

int MBIstftVitsModel::GetAudioOutput(RSState& state, float** out_data) {
  auto& s = static_cast<MBIstftVitsState&>(state);
  if (s.audio_read_cursor >= (int)s.audio_output.size()) {
    *out_data = nullptr;
    return 0;
  }
  *out_data = s.audio_output.data() + s.audio_read_cursor;
  int n = (int)s.audio_output.size() - s.audio_read_cursor;
  s.audio_read_cursor = (int)s.audio_output.size();
  return n;
}

// ---------------------------------------------------------------------
// Capture helper: store a host buffer that is in ggml [C,T] order
// (channel fastest) into the dump map in PyTorch [1,C,T] order.
// ---------------------------------------------------------------------
static void mbv_dump_ct(MBIstftVitsState& s, const std::string& name,
                        const float* data, int64_t C, int64_t T) {
  if (!s.capture) return;
  std::vector<float> out((size_t)C * T);
  for (int64_t t = 0; t < T; t++)
    for (int64_t c = 0; c < C; c++)
      out[(size_t)c * T + t] = data[(size_t)t * C + c];
  s.dumps[name] = std::move(out);
  s.dump_shapes[name] = {1, C, T};
}

static void mbv_dump_raw(MBIstftVitsState& s, const std::string& name,
                         const float* data, std::vector<int64_t> shape) {
  if (!s.capture) return;
  size_t n = 1;
  for (auto d : shape) n *= (size_t)d;
  s.dumps[name] = std::vector<float>(data, data + n);
  s.dump_shapes[name] = std::move(shape);
}

static std::vector<float> mbv_read(struct ggml_tensor* t) {
  std::vector<float> v(ggml_nelements(t));
  ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
  return v;
}

// =====================================================================
// Stage 1: TextEncoder + DurationPredictor  (one graph)
// =====================================================================
bool MBIstftVitsModel::RunEncoderAndDP(MBIstftVitsState& s,
                                       ggml_backend_sched_t sched) {
  const int T  = (int)s.phone_ids.size();
  const int C  = hparams_.hidden_channels;
  const int nh = hparams_.n_heads;
  const int hd = hparams_.k_channels;

  struct ggml_context* ctx0 = nullptr;
  struct ggml_cgraph* gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, MBV_MAX_NODES)) return false;
  g_mbv_pending.clear();

  struct ggml_tensor* pid = mbv_input_i32(ctx0, "phone_ids", s.phone_ids.data(), T);
  struct ggml_tensor* tid = mbv_input_i32(ctx0, "tone_ids",  s.tone_ids.data(),  T);
  struct ggml_tensor* lid = mbv_input_i32(ctx0, "lang_ids",  s.lang_ids.data(),  T);

  struct ggml_tensor* emb =
      ggml_add(ctx0, ggml_add(ctx0,
          ggml_get_rows(ctx0, Get("enc_p.emb_phone.weight"), pid),
          ggml_get_rows(ctx0, Get("enc_p.emb_tone.weight"), tid)),
          ggml_get_rows(ctx0, Get("enc_p.emb_lang.weight"), lid));
  emb = ggml_scale(ctx0, emb, sqrtf((float)C));  // (192, T)
  ggml_set_name(emb, "emb");
  ggml_set_output(emb);

  struct ggml_tensor* x = emb;
  std::vector<struct ggml_tensor*> attn_outs;
  const int ffn_pad = (hparams_.kernel_size - 1) / 2;

  for (int i = 0; i < hparams_.n_layers; i++) {
    const std::string ap = "enc_p.encoder.attn_layers." + std::to_string(i);
    const std::string n1 = "enc_p.encoder.norm_layers_1." + std::to_string(i);
    const std::string fp = "enc_p.encoder.ffn_layers." + std::to_string(i);
    const std::string n2 = "enc_p.encoder.norm_layers_2." + std::to_string(i);

    struct ggml_tensor* y = mbv_rel_mha(
        ctx0, x,
        Get(ap + ".conv_q.weight"), Get(ap + ".conv_q.bias"),
        Get(ap + ".conv_k.weight"), Get(ap + ".conv_k.bias"),
        Get(ap + ".conv_v.weight"), Get(ap + ".conv_v.bias"),
        Get(ap + ".conv_o.weight"), Get(ap + ".conv_o.bias"),
        Get(ap + ".emb_rel_k"), Get(ap + ".emb_rel_v"),
        nh, hd, T, hparams_.window_size, i);
    {
      char nb[32];
      snprintf(nb, sizeof(nb), "attn%d", i);
      ggml_set_name(y, nb);
      ggml_set_output(y);
      attn_outs.push_back(y);
    }
    x = mbv_layer_norm_ct(ctx0, ggml_add(ctx0, x, y),
                          Get(n1 + ".gamma"), Get(n1 + ".beta"));

    struct ggml_tensor* f = mbv_conv1d_ct(ctx0, x, Get(fp + ".conv_1.weight"),
                                          Get(fp + ".conv_1.bias"), ffn_pad, 1);
    f = ggml_relu(ctx0, f);
    f = mbv_conv1d_ct(ctx0, f, Get(fp + ".conv_2.weight"),
                      Get(fp + ".conv_2.bias"), ffn_pad, 1);
    x = mbv_layer_norm_ct(ctx0, ggml_add(ctx0, x, f),
                          Get(n2 + ".gamma"), Get(n2 + ".beta"));
  }
  ggml_set_name(x, "enc");
  ggml_set_output(x);

  // proj -> (384, T), split into m_p / logs_p
  struct ggml_tensor* stats = mbv_linear_ct(ctx0, x, Get("enc_p.proj.weight"),
                                            Get("enc_p.proj.bias"));
  struct ggml_tensor* m_p = ggml_cont(ctx0,
      ggml_view_2d(ctx0, stats, C, T, stats->nb[1], 0));
  struct ggml_tensor* logs_p = ggml_cont(ctx0,
      ggml_view_2d(ctx0, stats, C, T, stats->nb[1], (size_t)C * sizeof(float)));
  ggml_set_name(m_p, "m_p");     ggml_set_output(m_p);
  ggml_set_name(logs_p, "logs_p"); ggml_set_output(logs_p);

  // Duration predictor
  struct ggml_tensor* d = mbv_conv1d_ct(ctx0, x, Get("dp.conv_1.weight"),
                                        Get("dp.conv_1.bias"), 1, 1);
  d = ggml_relu(ctx0, d);
  d = mbv_layer_norm_ct(ctx0, d, Get("dp.norm_1.gamma"), Get("dp.norm_1.beta"));
  d = mbv_conv1d_ct(ctx0, d, Get("dp.conv_2.weight"), Get("dp.conv_2.bias"), 1, 1);
  d = ggml_relu(ctx0, d);
  d = mbv_layer_norm_ct(ctx0, d, Get("dp.norm_2.gamma"), Get("dp.norm_2.beta"));
  struct ggml_tensor* logw = mbv_linear_ct(ctx0, d, Get("dp.proj.weight"),
                                           Get("dp.proj.bias"));  // (1, T)
  ggml_set_name(logw, "logw");
  ggml_set_output(logw);

  ggml_build_forward_expand(gf, logw);
  ggml_build_forward_expand(gf, m_p);
  ggml_build_forward_expand(gf, logs_p);

  ggml_backend_sched_reset(sched);
  mbv_apply_sched_policy(sched, gf, getenv("MBISTFT_ENC_CPU") != nullptr, false);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("mbistft-vits: encoder graph alloc failed");
    ggml_free(ctx0);
    return false;
  }
  mbv_flush_inputs();
  ggml_backend_sched_graph_compute(sched, gf);

  s.T_text = T;
  {
    auto v = mbv_read(m_p);    s.m_p    = v; mbv_dump_ct(s, "m_p",    v.data(), C, T);
    auto w = mbv_read(logs_p); s.logs_p = w; mbv_dump_ct(s, "logs_p", w.data(), C, T);
    auto l = mbv_read(logw);   s.logw   = l;
    if (s.capture) mbv_dump_raw(s, "logw", l.data(), {1, 1, T});
  }
  if (s.capture) {
    auto e = mbv_read(emb);
    mbv_dump_ct(s, "emb", e.data(), C, T);
    for (int i = 0; i < (int)attn_outs.size(); i++) {
      auto a = mbv_read(attn_outs[i]);
      mbv_dump_ct(s, "attn" + std::to_string(i), a.data(), C, T);
    }
    auto en = mbv_read(x);
    mbv_dump_ct(s, "enc", en.data(), C, T);
  }

  ggml_free(ctx0);
  return true;
}

// =====================================================================
// Stage 2: flow (reverse).  flows = [RC0, Flip, RC2, Flip, RC4, Flip,
// RC6, Flip]; reversed order = Flip,RC6,Flip,RC4,Flip,RC2,Flip,RC0.
// =====================================================================
bool MBIstftVitsModel::RunFlowReverse(MBIstftVitsState& s,
                                      ggml_backend_sched_t sched) {
  const int C  = hparams_.inter_channels;   // 192
  const int Ch = C / 2;                     // 96
  const int Tf = s.T_frames;
  const int wn_pad = (hparams_.flow_kernel_size - 1) / 2;

  struct ggml_context* ctx0 = nullptr;
  struct ggml_cgraph* gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, MBV_MAX_NODES)) return false;
  g_mbv_pending.clear();

  struct ggml_tensor* z = mbv_input_f32(ctx0, "z_p", s.z_p.data(), C, Tf);
  z = ggml_reshape_2d(ctx0, z, C, Tf);

  std::vector<int32_t> rev(C);
  for (int i = 0; i < C; i++) rev[i] = C - 1 - i;
  struct ggml_tensor* rev_idx = mbv_input_i32(ctx0, "rev_idx", rev.data(), C);

  struct ggml_tensor* x = z;
  for (int fi = hparams_.n_flows - 1; fi >= 0; fi--) {
    x = mbv_flip_channels(ctx0, x, rev_idx);  // the Flip after coupling fi

    const std::string p = "flow.flows." + std::to_string(2 * fi);
    struct ggml_tensor* x0 = ggml_cont(ctx0,
        ggml_view_2d(ctx0, x, Ch, Tf, x->nb[1], 0));
    struct ggml_tensor* x1 = ggml_cont(ctx0,
        ggml_view_2d(ctx0, x, Ch, Tf, x->nb[1], (size_t)Ch * sizeof(float)));

    struct ggml_tensor* h = mbv_linear_ct(ctx0, x0, Get(p + ".pre.weight"),
                                          Get(p + ".pre.bias"));  // (192, Tf)
    // WN (4 layers, dilation 1, no cond)
    struct ggml_tensor* out_acc = nullptr;
    for (int j = 0; j < hparams_.flow_n_layers; j++) {
      const std::string ip = p + ".enc.in_layers." + std::to_string(j);
      const std::string rp = p + ".enc.res_skip_layers." + std::to_string(j);
      struct ggml_tensor* xin = mbv_conv1d_ct(ctx0, h, Get(ip + ".weight"),
                                              Get(ip + ".bias"), wn_pad, 1);  // (384, Tf)
      struct ggml_tensor* ta = ggml_tanh(ctx0, ggml_cont(ctx0,
          ggml_view_2d(ctx0, xin, C, Tf, xin->nb[1], 0)));
      struct ggml_tensor* sa = ggml_sigmoid(ctx0, ggml_cont(ctx0,
          ggml_view_2d(ctx0, xin, C, Tf, xin->nb[1], (size_t)C * sizeof(float))));
      struct ggml_tensor* acts = ggml_mul(ctx0, ta, sa);  // (192, Tf)
      struct ggml_tensor* rs = mbv_linear_ct(ctx0, acts, Get(rp + ".weight"),
                                             Get(rp + ".bias"));
      if (j < hparams_.flow_n_layers - 1) {
        struct ggml_tensor* res = ggml_cont(ctx0,
            ggml_view_2d(ctx0, rs, C, Tf, rs->nb[1], 0));
        struct ggml_tensor* skip = ggml_cont(ctx0,
            ggml_view_2d(ctx0, rs, C, Tf, rs->nb[1], (size_t)C * sizeof(float)));
        h = ggml_add(ctx0, h, res);
        out_acc = out_acc ? ggml_add(ctx0, out_acc, skip) : skip;
      } else {
        out_acc = out_acc ? ggml_add(ctx0, out_acc, rs) : rs;
      }
    }
    struct ggml_tensor* m = mbv_linear_ct(ctx0, out_acc, Get(p + ".post.weight"),
                                          Get(p + ".post.bias"));  // (96, Tf)
    struct ggml_tensor* x1n = ggml_sub(ctx0, x1, m);  // reverse, logs=0
    x = ggml_concat(ctx0, x0, x1n, /*dim=*/0);        // (192, Tf)
  }

  ggml_set_name(x, "z_flow");
  ggml_set_output(x);
  ggml_build_forward_expand(gf, x);

  ggml_backend_sched_reset(sched);
  mbv_apply_sched_policy(sched, gf, getenv("MBISTFT_FLOW_CPU") != nullptr, false);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("mbistft-vits: flow graph alloc failed");
    ggml_free(ctx0);
    return false;
  }
  mbv_flush_inputs();
  ggml_backend_sched_graph_compute(sched, gf);

  s.z = mbv_read(x);
  mbv_dump_ct(s, "z_flow", s.z.data(), C, Tf);

  ggml_free(ctx0);
  return true;
}

// =====================================================================
// Stage 3: Multiband iSTFT generator (decoder), [T,C] layout throughout.
// =====================================================================
bool MBIstftVitsModel::RunDecoder(MBIstftVitsState& s,
                                  ggml_backend_sched_t sched) {
  const int C  = hparams_.inter_channels;
  const int Tf = s.T_frames;
  const int n_fft = hparams_.gen_istft_n_fft;   // 16
  const int hop   = hparams_.gen_istft_hop;     // 4
  const int bins  = n_fft / 2 + 1;              // 9
  const int sub   = hparams_.subbands;          // 4
  const int num_kernels = (int)hparams_.resblock_kernel_sizes.size();

  struct ggml_context* ctx0 = nullptr;
  struct ggml_cgraph* gf = nullptr;
  if (!init_compute_ctx(&ctx0, &gf, MBV_MAX_NODES)) return false;
  g_mbv_pending.clear();

  // upload z transposed to [T,C]
  std::vector<float> z_tc((size_t)Tf * C);
  for (int t = 0; t < Tf; t++)
    for (int c = 0; c < C; c++)
      z_tc[(size_t)c * Tf + t] = s.z[(size_t)t * C + c];
  struct ggml_tensor* z = mbv_input_f32(ctx0, "z", z_tc.data(), Tf, C);
  z = ggml_reshape_2d(ctx0, z, Tf, C);

  struct ggml_tensor* x = mbv_conv1d_tc(ctx0, z, Get("dec.conv_pre.weight"),
                                        Get("dec.conv_pre.bias"), 3, 1);  // (Tf, 512)

  for (int i = 0; i < (int)hparams_.upsample_rates.size(); i++) {
    x = ggml_leaky_relu(ctx0, x, 0.1f, false);
    // ConvTranspose1d stride u, kernel k, padding (k-u)/2: run unpadded then trim.
    const int u = hparams_.upsample_rates[i];
    const int k = hparams_.upsample_kernel_sizes[i];
    const int trim = (k - u) / 2;
    const std::string up = "dec.ups." + std::to_string(i);
    struct ggml_tensor* w = Get(up + ".weight");   // (K, OC, IC)
    struct ggml_tensor* ct = mbv_convt1d(ctx0, w,
        ggml_cont(ctx0, x), u);                    // (OL_full, OC, 1)
    const int64_t OLf = ct->ne[0];
    const int64_t OC  = ct->ne[1];
    struct ggml_tensor* ct2 = ggml_reshape_2d(ctx0, ct, OLf, OC);
    struct ggml_tensor* trimmed = ggml_cont(ctx0,
        ggml_view_2d(ctx0, ct2, OLf - 2 * trim, OC, ct2->nb[1],
                     (size_t)trim * sizeof(float)));
    x = ggml_add(ctx0, trimmed,
                 ggml_reshape_2d(ctx0, Get(up + ".bias"), 1, OC));

    // 3 parallel resblocks, averaged
    struct ggml_tensor* xs = nullptr;
    for (int j = 0; j < num_kernels; j++) {
      const int rk = hparams_.resblock_kernel_sizes[j];
      const std::string rb = "dec.resblocks." + std::to_string(i * num_kernels + j);
      static const int dils[3] = {1, 3, 5};
      struct ggml_tensor* rx = x;
      for (int di = 0; di < 3; di++) {
        const int dil = dils[di];
        struct ggml_tensor* xt = ggml_leaky_relu(ctx0, rx, 0.1f, false);
        xt = mbv_conv1d_tc(ctx0, xt,
            Get(rb + ".convs1." + std::to_string(di) + ".weight"),
            Get(rb + ".convs1." + std::to_string(di) + ".bias"),
            (rk * dil - dil) / 2, dil);
        xt = ggml_leaky_relu(ctx0, xt, 0.1f, false);
        xt = mbv_conv1d_tc(ctx0, xt,
            Get(rb + ".convs2." + std::to_string(di) + ".weight"),
            Get(rb + ".convs2." + std::to_string(di) + ".bias"),
            (rk - 1) / 2, 1);
        rx = ggml_add(ctx0, rx, xt);
      }
      xs = xs ? ggml_add(ctx0, xs, rx) : rx;
    }
    x = ggml_scale(ctx0, xs, 1.0f / (float)num_kernels);
  }

  // NOTE: models.py uses F.leaky_relu(x) here — DEFAULT slope 0.01!
  x = ggml_leaky_relu(ctx0, x, 0.01f, false);

  // ReflectionPad1d((1,0)): prepend x[t=1] on the time axis (ne0)
  {
    struct ggml_tensor* col = ggml_cont(ctx0,
        ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], sizeof(float)));
    x = ggml_concat(ctx0, col, x, 0);  // (T+1, C)
  }

  x = mbv_conv1d_tc(ctx0, x, Get("dec.subband_conv_post.weight"),
                    Get("dec.subband_conv_post.bias"), 3, 1);  // (Tp, 72)
  const int64_t Tp = x->ne[0];

  // channel c = s*(n_fft+2) + k  ->  view as (Tp, n_fft+2, sub)
  struct ggml_tensor* x3 = ggml_reshape_3d(ctx0, ggml_cont(ctx0, x),
                                           Tp, n_fft + 2, sub);
  struct ggml_tensor* spec_pre = ggml_view_3d(ctx0, x3, Tp, bins, sub,
      x3->nb[1], x3->nb[2], 0);
  struct ggml_tensor* phase_pre = ggml_view_3d(ctx0, x3, Tp, bins, sub,
      x3->nb[1], x3->nb[2], (size_t)bins * x3->nb[1]);

  struct ggml_tensor* spec = ggml_exp(ctx0, ggml_cont(ctx0, spec_pre));
  struct ggml_tensor* phase = ggml_scale(ctx0,
      ggml_sin(ctx0, ggml_cont(ctx0, phase_pre)), (float)M_PI);

  struct ggml_tensor* real = ggml_mul(ctx0, spec, ggml_cos(ctx0, phase));
  struct ggml_tensor* imag = ggml_mul(ctx0, spec, ggml_sin(ctx0, phase));

  // permute (Tp, bins, sub) -> (bins, Tp, sub) for the irFFT matmul
  real = ggml_cont(ctx0, ggml_permute(ctx0, real, 1, 0, 2, 3));
  imag = ggml_cont(ctx0, ggml_permute(ctx0, imag, 1, 0, 2, 3));

  // irFFT as fixed matrices: frames[n,t,s] = sum_k real*C[k,n] + imag*S[k,n]
  std::vector<float> Cm((size_t)bins * n_fft), Sm((size_t)bins * n_fft);
  for (int n = 0; n < n_fft; n++) {
    for (int k = 0; k < bins; k++) {
      const float coef = (k == 0 || k == n_fft / 2) ? 1.0f : 2.0f;
      const double ang = 2.0 * M_PI * k * n / n_fft;
      Cm[(size_t)n * bins + k] = (float)( coef * cos(ang) / n_fft);
      Sm[(size_t)n * bins + k] = (float)(-coef * sin(ang) / n_fft);
    }
  }
  struct ggml_tensor* Cmat = mbv_input_f32(ctx0, "istft_C", Cm.data(), bins, n_fft);
  struct ggml_tensor* Smat = mbv_input_f32(ctx0, "istft_S", Sm.data(), bins, n_fft);
  Cmat = ggml_reshape_2d(ctx0, Cmat, bins, n_fft);
  Smat = ggml_reshape_2d(ctx0, Smat, bins, n_fft);

  struct ggml_tensor* frames = ggml_add(ctx0,
      ggml_mul_mat(ctx0, Cmat, real),
      ggml_mul_mat(ctx0, Smat, imag));         // (n_fft, Tp, sub)
  frames = ggml_mul(ctx0, frames, Get("istft.window"));

  // OLA via conv_transpose_1d with an identity kernel (n_fft, 1, n_fft)
  std::vector<float> eye((size_t)n_fft * n_fft, 0.0f);
  for (int i = 0; i < n_fft; i++) eye[(size_t)i * n_fft + i] = 1.0f;
  struct ggml_tensor* ola_k = mbv_input_f32(ctx0, "ola_kernel", eye.data(),
                                            n_fft, 1, n_fft);

  // window-envelope normalization: conv_transpose of ones with win^2
  std::vector<float> win2(n_fft);
  for (int i = 0; i < n_fft; i++) win2[i] = istft_window_[i] * istft_window_[i];
  struct ggml_tensor* env_k = mbv_input_f32(ctx0, "env_kernel", win2.data(),
                                            n_fft, 1, 1);
  std::vector<float> ones_v((size_t)Tp, 1.0f);
  struct ggml_tensor* ones = mbv_input_f32(ctx0, "ones", ones_v.data(), Tp, 1);
  ones = ggml_reshape_2d(ctx0, ones, Tp, 1);
  struct ggml_tensor* env = mbv_convt1d(ctx0, env_k, ones, hop);
  env = ggml_reshape_2d(ctx0, env, env->ne[0], 1);
  env = ggml_clamp(ctx0, env, 1e-9f, INFINITY);

  const int half = n_fft / 2;
  struct ggml_tensor* o_mb = nullptr;  // (Ls, sub)
  for (int sb = 0; sb < sub; sb++) {
    struct ggml_tensor* fs = ggml_view_2d(ctx0, frames, n_fft, Tp,
        frames->nb[1], (size_t)sb * frames->nb[2]);
    struct ggml_tensor* fst = ggml_cont(ctx0, ggml_transpose(ctx0, fs));  // (Tp, n_fft)
    struct ggml_tensor* y = mbv_convt1d(ctx0, ola_k, fst, hop);
    y = ggml_reshape_2d(ctx0, y, y->ne[0], 1);
    y = ggml_div(ctx0, y, env);
    // center=True trim: n_fft/2 from both ends
    struct ggml_tensor* yt = ggml_cont(ctx0,
        ggml_view_2d(ctx0, y, y->ne[0] - 2 * half, 1, y->nb[1],
                     (size_t)half * sizeof(float)));
    o_mb = o_mb ? ggml_concat(ctx0, o_mb, yt, 1) : yt;
  }
  ggml_set_name(o_mb, "o_mb");
  ggml_set_output(o_mb);

  // PQMF synthesis
  struct ggml_tensor* ud = ggml_scale(ctx0, Get("pqmf.updown_filter"), (float)sub);
  struct ggml_tensor* up = mbv_convt1d(ctx0, ud,
      ggml_cont(ctx0, o_mb), sub);  // (sub*Ls, sub, 1)
  up = ggml_reshape_2d(ctx0, up, up->ne[0], up->ne[1]);
  // gguf stores the baked filter as torch [4,1,63] (ne 63,1,4); the synthesis
  // conv wants [OC=1, IC=4, K=63] (ne 63,4,1) — identical memory, reshape only.
  struct ggml_tensor* syn = Get("pqmf.synthesis_filter");
  syn = ggml_reshape_3d(ctx0, syn, syn->ne[0], sub, 1);    // (63, 4, 1)
  const int taps_half = (int)(syn->ne[0] - 1) / 2;         // 31
  struct ggml_tensor* wav = mbv_conv1d_tc(ctx0, up, syn, nullptr, taps_half, 1);
  wav = ggml_reshape_1d(ctx0, wav, wav->ne[0]);
  ggml_set_name(wav, "wav");
  ggml_set_output(wav);

  ggml_build_forward_expand(gf, wav);

  ggml_backend_sched_reset(sched);
  mbv_apply_sched_policy(sched, gf, getenv("MBISTFT_DEC_CPU") != nullptr,
                         getenv("MBISTFT_DEC_CT_CPU") != nullptr);
  if (!ggml_backend_sched_alloc_graph(sched, gf)) {
    RS_LOG_ERR("mbistft-vits: decoder graph alloc failed");
    ggml_free(ctx0);
    return false;
  }
  mbv_flush_inputs();
  ggml_backend_sched_graph_compute(sched, gf);

  s.audio_output = mbv_read(wav);
  s.audio_read_cursor = 0;
  if (s.capture) {
    auto mb = mbv_read(o_mb);  // (Ls, sub): flat = s*Ls + l == torch [1,sub,Ls]
    mbv_dump_raw(s, "o_mb", mb.data(), {1, sub, o_mb->ne[0]});
    mbv_dump_raw(s, "wav", s.audio_output.data(),
                 {1, 1, (int64_t)s.audio_output.size()});
  }

  ggml_free(ctx0);
  return true;
}

// =====================================================================
// Encode / Decode
// =====================================================================
bool MBIstftVitsModel::Encode(const std::vector<float>& input_frames,
                              RSState& state, ggml_backend_sched_t sched) {
  (void)input_frames;
  auto& s = static_cast<MBIstftVitsState&>(state);
  if (s.phone_ids.empty() || s.phone_ids.size() != s.tone_ids.size() ||
      s.phone_ids.size() != s.lang_ids.size()) {
    RS_LOG_ERR("mbistft-vits: phone/tone/lang ids not set (or length mismatch)");
    return false;
  }

  if (!RunEncoderAndDP(s, sched)) return false;

  // Length regulation on the host: w = exp(logw)*length_scale, w_ceil = ceil(w)
  const int C = hparams_.hidden_channels;
  const int T = s.T_text;
  s.w_ceil.resize(T);
  int total = 0;
  for (int t = 0; t < T; t++) {
    float w = expf(s.logw[t]) * s.length_scale;
    int wc = (int)ceilf(w);
    if (wc < 0) wc = 0;
    s.w_ceil[t] = wc;
    total += wc;
  }
  if (total < 1) total = 1;
  s.T_frames = total;

  if (s.capture) {
    std::vector<float> wcf(T);
    for (int t = 0; t < T; t++) wcf[t] = (float)s.w_ceil[t];
    mbv_dump_raw(s, "w_ceil", wcf.data(), {1, 1, T});
  }

  // teacher-forced durations, if provided
  const std::vector<int32_t>* dur = &s.w_ceil;
  if ((int)s.duration_override.size() == T) {
    dur = &s.duration_override;
    total = 0;
    for (int t = 0; t < T; t++) total += s.duration_override[t];
    if (total < 1) total = 1;
    s.T_frames = total;
  }

  // expand m_p / logs_p to frames ([C,T] layout: memcpy per frame)
  std::vector<float> m_p_exp((size_t)C * total), logs_p_exp((size_t)C * total);
  int fr = 0;
  for (int t = 0; t < T; t++) {
    for (int d = 0; d < (*dur)[t] && fr < total; d++, fr++) {
      std::memcpy(&m_p_exp[(size_t)fr * C], &s.m_p[(size_t)t * C], C * sizeof(float));
      std::memcpy(&logs_p_exp[(size_t)fr * C], &s.logs_p[(size_t)t * C], C * sizeof(float));
    }
  }
  // pad tail (only when total was clamped up to 1) with zeros — already zeroed.
  mbv_dump_ct(s, "m_p_exp", m_p_exp.data(), C, total);
  mbv_dump_ct(s, "logs_p_exp", logs_p_exp.data(), C, total);

  // z_p = m_p_exp + randn * exp(logs_p_exp) * noise_scale
  s.z_p = m_p_exp;
  if (s.noise_scale > 0.0f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < s.z_p.size(); i++)
      s.z_p[i] += dist(rng) * expf(logs_p_exp[i]) * s.noise_scale;
  }
  mbv_dump_ct(s, "z_p", s.z_p.data(), C, total);

  return RunFlowReverse(s, sched);
}

bool MBIstftVitsModel::Decode(RSState& state, ggml_backend_sched_t sched) {
  auto& s = static_cast<MBIstftVitsState&>(state);
  if (s.z.empty() || s.T_frames <= 0) {
    RS_LOG_ERR("mbistft-vits: Decode called before Encode");
    return false;
  }
  return RunDecoder(s, sched);
}

// =====================================================================
// Static registration
// =====================================================================
namespace {
struct MBIstftVitsRegistrar {
  MBIstftVitsRegistrar() {
    rs_register_model_arch("mbistft-vits", []() {
      return std::make_shared<MBIstftVitsModel>();
    });
  }
} global_mbistft_vits_reg;
}  // namespace
