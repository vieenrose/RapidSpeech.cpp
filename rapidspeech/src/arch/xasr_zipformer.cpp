// X-ASR Zipformer2 encoder graph construction (offline full-context mode).
//
// Layout convention: sequence tensors are [C, T] (ne0 = channels innermost),
// batch = 1. PyTorch reference: scripts/xasr_ref/zipformer.py.
//
// Streaming (chunked, cached) forward reuses these builders in M4 by passing
// cache tensors; for now only the offline path is implemented.

#include "arch/xasr.h"
#include "ggml.h"
#include "utils/rs_log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// host-side precomputation
// ---------------------------------------------------------------------------

// CompactRelPositionalEncoding: embeddings for relative positions
// [-(T+left-1) .. (T-1)], ascending; column-major into [pos_dim, S2].
// pe[:, j] corresponds to position j - (T + left - 1).
void xasr_compute_pos_emb(int T, int left, int pos_dim,
                          std::vector<float> &out) {
  const int S2 = left + 2 * T - 1;
  out.resize((size_t)pos_dim * S2);
  const double cl = std::sqrt((double)pos_dim); // compression_length
  const double length_scale = 1.0 * pos_dim / (2.0 * M_PI);
  for (int j = 0; j < S2; j++) {
    const double x = (double)(j - (T + left - 1));
    const double sign = (x > 0) - (x < 0);
    const double x_c = cl * sign * (std::log(std::fabs(x) + cl) - std::log(cl));
    const double x_atan = std::atan(x_c / length_scale);
    float *col = out.data() + (size_t)j * pos_dim;
    for (int i = 0; i < pos_dim / 2; i++) {
      const double f = x_atan * (i + 1);
      col[2 * i] = (float)std::cos(f);
      col[2 * i + 1] = (float)std::sin(f);
    }
    col[pos_dim - 1] = 1.0f; // bias term
  }
}

// ChunkCausalDepthwiseConv1d chunkwise scale for a single chunk of length T
// (offline: T = whole sequence at this stack's rate).
// edge_scale host layout: [2, C, K] (left, right). Output [C, T] col-major
// into ggml [C, T] (ne0 = C) => out[t * C + c].
void xasr_compute_chunk_scale(const std::vector<float> &edge_scale, int C,
                              int K, int T, std::vector<float> &out) {
  out.assign((size_t)C * T, 1.0f);
  const float *left = edge_scale.data();
  const float *right = edge_scale.data() + (size_t)C * K;
  for (int c = 0; c < C; c++) {
    if (T < K) {
      // left_edge[:, :T], right_edge[:, -T:]
      for (int t = 0; t < T; t++) {
        out[(size_t)t * C + c] += left[c * K + t] + right[c * K + (K - T) + t];
      }
    } else {
      for (int t = 0; t < K; t++)
        out[(size_t)t * C + c] += left[c * K + t];
      for (int t = 0; t < K; t++)
        out[(size_t)(T - K + t) * C + c] += right[c * K + t];
    }
  }
}

// ---------------------------------------------------------------------------
// graph builder
// ---------------------------------------------------------------------------

struct XASRGraphBuilder {
  XASRModel &m;
  struct ggml_context *ctx;
  int T_fbank; // input fbank frames

  // staged inputs: tensor + host data to upload after sched alloc
  std::vector<std::pair<struct ggml_tensor *, std::vector<float>>> staged;
  // named intermediates for parity dumps (marked as outputs)
  std::vector<std::pair<std::string, struct ggml_tensor *>> dumps;

  struct ggml_tensor *fbank_in = nullptr;

  // ---- streaming per-layer cache plumbing ----
  struct LayerStream {
    int left = 0; // left context at this stack's rate
    struct ggml_tensor *key_in = nullptr, *nonlin_in = nullptr,
                       *val1_in = nullptr, *val2_in = nullptr,
                       *conv1_in = nullptr, *conv2_in = nullptr;
    struct ggml_tensor *key_out = nullptr, *nonlin_out = nullptr,
                       *val1_out = nullptr, *val2_out = nullptr,
                       *conv1_out = nullptr, *conv2_out = nullptr;
    struct ggml_tensor *mask = nullptr; // additive [left+T] (0 / -1000)
  };
  std::vector<LayerStream> lstream; // one per encoder layer when streaming
  struct ggml_tensor *embed_cache_in = nullptr, *embed_cache_out = nullptr;
  std::vector<struct ggml_tensor *> mask_in; // per stack (streaming)
  std::vector<int> mask_ds;                  // per stack ds (streaming)

  XASRGraphBuilder(XASRModel &model, struct ggml_context *c, int T)
      : m(model), ctx(c), T_fbank(T) {}

  void dump(const char *name, struct ggml_tensor *t) {
    ggml_set_output(t);
    dumps.emplace_back(name, t);
  }

  struct ggml_tensor *stage_input(std::vector<float> &&data, int64_t ne0,
                                  int64_t ne1) {
    struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, ne1);
    ggml_set_input(t);
    staged.emplace_back(t, std::move(data));
    return t;
  }

  // --- primitives -----------------------------------------------------------

  struct ggml_tensor *swoosh_l(struct ggml_tensor *x) {
    struct ggml_tensor *sp =
        ggml_softplus(ctx, ggml_scale_bias(ctx, x, 1.0f, -4.0f));
    return ggml_add(ctx, sp, ggml_scale_bias(ctx, x, -0.08f, -0.035f));
  }

  struct ggml_tensor *swoosh_r(struct ggml_tensor *x) {
    struct ggml_tensor *sp =
        ggml_softplus(ctx, ggml_scale_bias(ctx, x, 1.0f, -1.0f));
    return ggml_add(ctx, sp, ggml_scale_bias(ctx, x, -0.08f, -0.313261687f));
  }

  // BiasNorm: y = x * (mean((x - bias)^2, ch)^-0.5 * exp(log_scale)).
  // x: [C, T...]; bias: [C]; scale = exp(log_scale) host scalar.
  struct ggml_tensor *bias_norm(struct ggml_tensor *x,
                                struct ggml_tensor *bias, float scale) {
    struct ggml_tensor *xb = ggml_sub(ctx, x, bias);
    struct ggml_tensor *ms = ggml_mean(ctx, ggml_sqr(ctx, xb)); // [1, T]
    return ggml_div(ctx, ggml_scale(ctx, x, scale), ggml_sqrt(ctx, ms));
  }

  // F.linear: W [in, out] (ggml view of PT [out, in]), x [in, T] -> [out, T]
  struct ggml_tensor *linear(struct ggml_tensor *w, struct ggml_tensor *b,
                             struct ggml_tensor *x) {
    struct ggml_tensor *y = ggml_mul_mat(ctx, w, x);
    return b ? ggml_add(ctx, y, b) : y;
  }

  // y = orig + (x - orig) * scale, scale [C]
  struct ggml_tensor *bypass(struct ggml_tensor *orig, struct ggml_tensor *x,
                             struct ggml_tensor *scale) {
    return ggml_add(ctx, orig,
                    ggml_mul(ctx, ggml_sub(ctx, x, orig), scale));
  }

  // depthwise conv1d in f32 (mirrors ggml_conv_1d_dw but keeps im2col f32).
  // kernel [K, 1, C], data [T, C] (ne0 = time). returns [T_out, C].
  struct ggml_tensor *conv_dw_f32(struct ggml_tensor *kernel,
                                  struct ggml_tensor *data_tc, int pad) {
    struct ggml_tensor *b4 = ggml_reshape_4d(ctx, data_tc, data_tc->ne[0], 1,
                                             data_tc->ne[1], 1);
    struct ggml_tensor *im = ggml_im2col(ctx, kernel, b4, 1, 0, pad, 0, 1, 0,
                                         false, GGML_TYPE_F32);
    struct ggml_tensor *y = ggml_mul_mat(ctx, im, kernel); // [T_out, 1, C]
    return ggml_reshape_2d(ctx, y, y->ne[0], y->ne[2]);
  }

  // depthwise conv2d in f32 (mirrors ggml_conv_2d_dw but keeps im2col f32 so
  // Metal doesn't hit the missing f32xf16 matmul kernel).
  // a [KW, KH, 1, C], b [W, H, C, N] -> [OW, OH, C, N]
  struct ggml_tensor *conv_dw_2d_f32(struct ggml_tensor *a,
                                     struct ggml_tensor *b, int s0, int s1,
                                     int p0, int p1) {
    struct ggml_tensor *new_a =
        ggml_reshape_4d(ctx, a, a->ne[0], a->ne[1], 1, a->ne[2] * a->ne[3]);
    struct ggml_tensor *im = ggml_im2col(
        ctx, new_a,
        ggml_reshape_4d(ctx, b, b->ne[0], b->ne[1], 1, b->ne[2] * b->ne[3]),
        s0, s1, p0, p1, 1, 1, true, GGML_TYPE_F32);
    struct ggml_tensor *new_b = ggml_reshape_4d(
        ctx, im, im->ne[0], im->ne[2] * im->ne[1], b->ne[2], b->ne[3]);
    new_a = ggml_reshape_4d(ctx, new_a, new_a->ne[0] * new_a->ne[1],
                            new_a->ne[2], new_a->ne[3], 1);
    struct ggml_tensor *y = ggml_mul_mat(ctx, new_a, new_b);
    return ggml_reshape_4d(ctx, y, im->ne[1], im->ne[2], b->ne[2], b->ne[3]);
  }

  // --- encoder_embed ---------------------------------------------------------
  // fbank [80, T] -> [dim0, T'] with T' = (T - 7) / 2 (offline, SAME-padded
  // ConvNeXt in time) or (T - 7) / 2 - 3 (streaming, cached left pad).
  struct ggml_tensor *build_embed(struct ggml_tensor *fbank, bool streaming) {
    const auto &e = m.embed_;
    // -> image [W=80(freq), H=T(time), C=1, N=1]
    struct ggml_tensor *x =
        ggml_reshape_4d(ctx, fbank, fbank->ne[0], fbank->ne[1], 1, 1);
    // conv0: k3, pad (freq 1, time 0)
    x = ggml_conv_2d(ctx, e.conv0_w, x, 1, 1, 1, 0, 1, 1);
    x = ggml_add(ctx, x, ggml_reshape_3d(ctx, e.conv0_b, 1, 1,
                                         e.conv0_b->ne[0]));
    x = swoosh_r(x);
    // conv4: k3 stride 2, no pad
    x = ggml_conv_2d(ctx, e.conv4_w, x, 2, 2, 0, 0, 1, 1);
    x = ggml_add(ctx, x, ggml_reshape_3d(ctx, e.conv4_b, 1, 1,
                                         e.conv4_b->ne[0]));
    x = swoosh_r(x);
    // conv7: k3 stride (time 1, freq 2)
    x = ggml_conv_2d(ctx, e.conv7_w, x, 2, 1, 0, 0, 1, 1);
    x = ggml_add(ctx, x, ggml_reshape_3d(ctx, e.conv7_b, 1, 1,
                                         e.conv7_b->ne[0]));
    x = swoosh_r(x);
    // x: [F=19, T', C=128, 1]

    // ConvNeXt: bypass + pw2(SwooshL(pw1(dw(x))))
    struct ggml_tensor *byp = x;
    struct ggml_tensor *y;
    if (streaming) {
      // bypass = first (T1 - 3) frames; dw conv over [cache, x] with no time
      // padding; new cache = frames [T_out, T_out+3) of the concatenation.
      const int64_t T1 = x->ne[1];
      const int64_t T_out = T1 - 3;
      byp = ggml_cont(ctx, ggml_view_3d(ctx, x, x->ne[0], T_out, x->ne[2],
                                        x->nb[1], x->nb[2], 0));
      struct ggml_tensor *xcat = ggml_concat(ctx, embed_cache_in, x, 1);
      struct ggml_tensor *nc = ggml_cont(
          ctx, ggml_view_3d(ctx, xcat, xcat->ne[0], 3, xcat->ne[2],
                            xcat->nb[1], xcat->nb[2],
                            (size_t)T_out * xcat->nb[1]));
      ggml_set_output(nc);
      embed_cache_out = nc;
      y = conv_dw_2d_f32(e.cnx_dw_w, xcat, 1, 1, 3, 0); // freq pad only
    } else {
      y = conv_dw_2d_f32(e.cnx_dw_w, x, 1, 1, 3, 3);
    }
    y = ggml_add(ctx, y, ggml_reshape_3d(ctx, e.cnx_dw_b, 1, 1,
                                         e.cnx_dw_b->ne[0]));
    // pointwise 1x1 via mul_mat over channel dim: [F,T,C] -> [C, F*T]
    const int64_t F = y->ne[0], Tp = y->ne[1], C = y->ne[2];
    struct ggml_tensor *y2 =
        ggml_cont(ctx, ggml_permute(ctx, y, 1, 2, 0, 3)); // [C, F, T]
    y2 = ggml_reshape_2d(ctx, y2, C, F * Tp);
    struct ggml_tensor *w1 =
        ggml_reshape_2d(ctx, e.cnx_pw1_w, C, e.cnx_pw1_w->ne[3]);
    y2 = ggml_add(ctx, ggml_mul_mat(ctx, w1, y2), e.cnx_pw1_b); // [3C, F*T]
    y2 = swoosh_l(y2);
    struct ggml_tensor *w2 =
        ggml_reshape_2d(ctx, e.cnx_pw2_w, e.cnx_pw2_w->ne[2], C);
    y2 = ggml_add(ctx, ggml_mul_mat(ctx, w2, y2), e.cnx_pw2_b); // [C, F*T]
    // back to [F, T, C]: y2 [C, F, T] -> permute
    y2 = ggml_reshape_3d(ctx, y2, C, F, Tp);
    y2 = ggml_cont(ctx, ggml_permute(ctx, y2, 2, 0, 1, 3)); // [F, T, C]
    x = ggml_add(ctx, byp, y2);

    // flatten: PT (b, t, c*f) with f fastest -> need [F, C, T] then 2D
    struct ggml_tensor *xf =
        ggml_cont(ctx, ggml_permute(ctx, x, 0, 2, 1, 3)); // [F, C, T]
    xf = ggml_reshape_2d(ctx, xf, F * C, Tp);
    // out linear + BiasNorm
    struct ggml_tensor *out = linear(m.embed_.out_w, m.embed_.out_b, xf);
    out = bias_norm(out, m.embed_.out_norm_bias, m.embed_.out_norm_scale);
    return out; // [dim0, T']
  }

  // --- attention weights (shared) -------------------------------------------
  // x [d, T]; returns attn_weights [k_len, T_tgt, h]; k_len = T (+left when
  // streaming). ls != null => streaming: concat cached key, emit new cache.
  //
  // Copy-avoidance: q/k/p are strided views straight into the in_proj output
  // (rows stay contiguous, which is all mul_mat/binary ops require), and the
  // rel-shift is a strided view of the pos-score matmul output. The only
  // ggml_cont left is the streaming k before concat.
  struct ggml_tensor *build_attn_weights(const XASRLayer &L, int si,
                                         struct ggml_tensor *x,
                                         struct ggml_tensor *pos_emb,
                                         LayerStream *ls) {
    const int h = m.hp_.num_heads[si];
    const int qhd = m.hp_.query_head_dim[si];
    const int phd = m.hp_.pos_head_dim[si];
    const int64_t T = x->ne[1];
    const int64_t S2 = pos_emb->ne[1]; // left + 2T - 1

    struct ggml_tensor *qkp =
        linear(L.attn_in_proj_w, L.attn_in_proj_b, x); // [(2q+p)h, T]
    const size_t es = ggml_element_size(qkp);

    // strided per-head view into qkp: [hd, h, T] -> permute -> [hd, T, h]
    auto head_view = [&](int hd, size_t off) {
      struct ggml_tensor *v3 = ggml_view_3d(
          ctx, qkp, hd, h, T, (size_t)hd * es, qkp->nb[1], off);
      return ggml_permute(ctx, v3, 0, 2, 1, 3); // rows contiguous
    };
    struct ggml_tensor *qh = head_view(qhd, 0);
    struct ggml_tensor *ph = head_view(phd, (size_t)2 * qhd * h * es);

    struct ggml_tensor *kh;
    int64_t k_len = T;
    if (ls) {
      struct ggml_tensor *k = ggml_cont(
          ctx, ggml_view_2d(ctx, qkp, qhd * h, T, qkp->nb[1],
                            (size_t)qhd * h * es));
      struct ggml_tensor *k_full = ggml_concat(ctx, ls->key_in, k, 1);
      k_len = ls->left + T;
      // new cache = last `left` columns (contiguous view, no copy)
      // new cache = last `left` columns. MUST be materialized (ggml_cont):
      // it is read back after compute to roll the cache into the next chunk,
      // so it cannot alias the concat buffer (the allocator may reuse it).
      struct ggml_tensor *nk = ggml_cont(
          ctx, ggml_view_2d(ctx, k_full, qhd * h, ls->left, k_full->nb[1],
                            (size_t)T * k_full->nb[1]));
      ggml_set_output(nk);
      ls->key_out = nk;
      struct ggml_tensor *k3 =
          ggml_reshape_3d(ctx, k_full, qhd, h, k_len);
      kh = ggml_permute(ctx, k3, 0, 2, 1, 3); // [qhd, k_len, h]
    } else {
      kh = head_view(qhd, (size_t)qhd * h * es);
    }

    // scores [k_len, T_tgt, h]
    struct ggml_tensor *scores = ggml_mul_mat(ctx, kh, qh);

    // pos_scores: linear_pos(pos_emb) [phd*h, S2] -> [phd, S2, h]
    struct ggml_tensor *pe = ggml_mul_mat(ctx, L.linear_pos_w, pos_emb);
    pe = ggml_permute(ctx, ggml_reshape_3d(ctx, pe, phd, h, S2), 0, 2, 1, 3);
    struct ggml_tensor *ps = ggml_mul_mat(ctx, pe, ph); // [S2, T_tgt, h]
    // rel->abs shift: out[j, t, hh] = ps[(T-1) - t + j, t, hh], j < k_len
    struct ggml_tensor *ps_shift = ggml_view_3d(
        ctx, ps, k_len, T, h, (S2 - 1) * es, ps->nb[2], (size_t)(T - 1) * es);
    scores = ggml_add(ctx, scores, ps_shift);

    if (ls && ls->mask) {
      // fused mask + softmax; mask [k_len, T] broadcasts over heads
      return ggml_soft_max_ext(ctx, scores, ls->mask, 1.0f, 0.0f);
    }
    return ggml_soft_max(ctx, scores); // over ne0 = src
  }

  // --- SelfAttention (value path) --------------------------------------------
  struct ggml_tensor *build_self_attn(const XASRLayer &L, int si,
                                      struct ggml_tensor *x,
                                      struct ggml_tensor *attn,
                                      struct ggml_tensor *in_w,
                                      struct ggml_tensor *in_b,
                                      struct ggml_tensor *out_w,
                                      struct ggml_tensor *out_b,
                                      LayerStream *ls,
                                      struct ggml_tensor *val_cache,
                                      struct ggml_tensor **val_cache_out) {
    const int h = m.hp_.num_heads[si];
    const int vhd = m.hp_.value_head_dim[si];
    const int64_t T = x->ne[1];
    struct ggml_tensor *v = linear(in_w, in_b, x); // [vh*h, T]
    int64_t k_len = T;
    if (ls) {
      v = ggml_concat(ctx, val_cache, v, 1); // [vd, left+T]
      k_len = ls->left + T;
      // materialized cache slice (read back post-compute; see key_out note)
      struct ggml_tensor *nv = ggml_cont(
          ctx, ggml_view_2d(ctx, v, (int64_t)vhd * h, ls->left, v->nb[1],
                            (size_t)T * v->nb[1]));
      ggml_set_output(nv);
      *val_cache_out = nv;
    }
    // v [vd, k_len] -> vt [k_len, vhd, h] (single transpose copy; mul_mat
    // needs the contraction dim contiguous on both operands)
    struct ggml_tensor *v3 = ggml_reshape_3d(ctx, v, vhd, h, k_len);
    struct ggml_tensor *vt =
        ggml_cont(ctx, ggml_permute(ctx, v3, 1, 2, 0, 3)); // [k_len, vhd, h]
    struct ggml_tensor *o = ggml_mul_mat(ctx, vt, attn);   // [vhd, T_tgt, h]
    o = ggml_cont(ctx, ggml_permute(ctx, o, 0, 2, 1, 3));  // [vhd, h, T]
    o = ggml_reshape_2d(ctx, o, (int64_t)vhd * h, T);
    return linear(out_w, out_b, o);
  }

  // --- NonlinAttention --------------------------------------------------------
  struct ggml_tensor *build_nonlin_attn(const XASRLayer &L, int si,
                                        struct ggml_tensor *x,
                                        struct ggml_tensor *attn,
                                        LayerStream *ls) {
    const int64_t T = x->ne[1];
    const int d = m.hp_.encoder_dim[si];
    const int hc = 3 * d / 4;
    struct ggml_tensor *pr = linear(L.na_in_w, L.na_in_b, x); // [3hc, T]
    const size_t es = ggml_element_size(pr);
    // s/v/y stay strided views (rows contiguous suffices for unary/binary)
    struct ggml_tensor *s =
        ggml_view_2d(ctx, pr, hc, T, pr->nb[1], 0);
    struct ggml_tensor *v =
        ggml_view_2d(ctx, pr, hc, T, pr->nb[1], (size_t)hc * es);
    struct ggml_tensor *y =
        ggml_view_2d(ctx, pr, hc, T, pr->nb[1], (size_t)2 * hc * es);
    v = ggml_mul(ctx, v, ggml_tanh(ctx, s)); // contiguous result
    int64_t k_len = T;
    if (ls) {
      v = ggml_concat(ctx, ls->nonlin_in, v, 1); // [hc, left+T]
      k_len = ls->left + T;
      // materialized cache slice (read back post-compute; see key_out note)
      struct ggml_tensor *nv = ggml_cont(
          ctx, ggml_view_2d(ctx, v, hc, ls->left, v->nb[1],
                            (size_t)T * v->nb[1]));
      ggml_set_output(nv);
      ls->nonlin_out = nv;
    }
    // attend with head 0 only: attn0 [k_len, T_tgt] (contiguous slice)
    struct ggml_tensor *attn0 =
        ggml_view_2d(ctx, attn, attn->ne[0], attn->ne[1], attn->nb[1], 0);
    struct ggml_tensor *vt =
        ggml_cont(ctx, ggml_transpose(ctx, v)); // [k_len, hc]
    struct ggml_tensor *o = ggml_mul_mat(ctx, vt, attn0); // [hc, T_tgt]
    o = ggml_mul(ctx, o, y);
    return linear(L.na_out_w, L.na_out_b, o);
  }

  // --- ConvolutionModule ------------------------------------------------------
  // offline: chunk = whole sequence, zero left pad;
  // streaming: cached left context replaces the causal pad, chunkwise path
  // sees only new frames with the chunk-local (cropped) edge scale.
  struct ggml_tensor *build_conv_module(const XASRLayer::XASRConv &cv, int si,
                                        struct ggml_tensor *x,
                                        LayerStream *ls,
                                        struct ggml_tensor *conv_cache,
                                        struct ggml_tensor **conv_cache_out) {
    const int d = m.hp_.encoder_dim[si];
    const int K = m.hp_.cnn_kernel[si];
    const int64_t T = x->ne[1];
    struct ggml_tensor *pr = linear(cv.in_w, cv.in_b, x); // [2d, T]
    const size_t es = ggml_element_size(pr);
    struct ggml_tensor *v =
        ggml_cont(ctx, ggml_view_2d(ctx, pr, d, T, pr->nb[1], 0));
    struct ggml_tensor *s = ggml_cont(
        ctx, ggml_view_2d(ctx, pr, d, T, pr->nb[1], (size_t)d * es));
    v = ggml_mul(ctx, v, ggml_sigmoid(ctx, s)); // [d, T]

    // -> [T, d] for depthwise convs
    struct ggml_tensor *vt = ggml_cont(ctx, ggml_transpose(ctx, v)); // [T, d]

    struct ggml_tensor *vpad; // [K/2 + T, d]
    if (ls) {
      vpad = ggml_concat(ctx, conv_cache, vt, 0); // cache [K/2, d]
      struct ggml_tensor *nc = ggml_cont(
          ctx, ggml_view_2d(ctx, vpad, K / 2, d, vpad->nb[1],
                            (size_t)T * ggml_element_size(vpad)));
      ggml_set_output(nc);
      *conv_cache_out = nc;
    } else {
      vpad = ggml_pad_ext(ctx, vt, K / 2, 0, 0, 0, 0, 0, 0, 0);
    }
    struct ggml_tensor *yc = conv_dw_f32(cv.causal_w, vpad, 0); // [T, d]
    yc = ggml_add(ctx, yc, ggml_reshape_2d(ctx, cv.causal_b, 1,
                                           cv.causal_b->ne[0]));

    // chunkwise path: SAME padding K/2 over the NEW frames only
    struct ggml_tensor *yk = conv_dw_f32(cv.chunk_w, vt, K / 2); // [T, d]
    yk = ggml_add(ctx, yk, ggml_reshape_2d(ctx, cv.chunk_b, 1,
                                           cv.chunk_b->ne[0]));
    struct ggml_tensor *yk_dt =
        ggml_cont(ctx, ggml_transpose(ctx, yk)); // [d, T]
    std::vector<float> scale_host;
    xasr_compute_chunk_scale(cv.edge_scale, d, K, (int)T, scale_host);
    struct ggml_tensor *scale_t = stage_input(std::move(scale_host), d, T);
    yk_dt = ggml_mul(ctx, yk_dt, scale_t);

    struct ggml_tensor *yc_dt = ggml_cont(ctx, ggml_transpose(ctx, yc));
    struct ggml_tensor *y = ggml_add(ctx, yk_dt, yc_dt); // [d, T]

    // out_proj = SwooshR + Linear
    return linear(cv.out_w, cv.out_b, swoosh_r(y));
  }

  struct ggml_tensor *build_ff(struct ggml_tensor *in_w,
                               struct ggml_tensor *in_b,
                               struct ggml_tensor *out_w,
                               struct ggml_tensor *out_b,
                               struct ggml_tensor *x) {
    return linear(out_w, out_b, swoosh_l(linear(in_w, in_b, x)));
  }

  // --- one encoder layer ------------------------------------------------------
  struct ggml_tensor *build_layer(const XASRLayer &L, int si,
                                  struct ggml_tensor *x,
                                  struct ggml_tensor *pos_emb,
                                  LayerStream *ls) {
    struct ggml_tensor *orig = x;
    struct ggml_tensor *attn = build_attn_weights(L, si, x, pos_emb, ls);

    x = ggml_add(ctx, x, build_ff(L.ff1_in_w, L.ff1_in_b, L.ff1_out_w,
                                  L.ff1_out_b, x));
    x = ggml_add(ctx, x, build_nonlin_attn(L, si, x, attn, ls));
    x = ggml_add(ctx, x,
                 build_self_attn(L, si, x, attn, L.attn1_in_w, L.attn1_in_b,
                                 L.attn1_out_w, L.attn1_out_b, ls,
                                 ls ? ls->val1_in : nullptr,
                                 ls ? &ls->val1_out : nullptr));
    x = ggml_add(ctx, x, build_conv_module(L.conv1, si, x, ls,
                                           ls ? ls->conv1_in : nullptr,
                                           ls ? &ls->conv1_out : nullptr));
    x = ggml_add(ctx, x, build_ff(L.ff2_in_w, L.ff2_in_b, L.ff2_out_w,
                                  L.ff2_out_b, x));
    x = bypass(orig, x, L.bypass_mid_scale);
    x = ggml_add(ctx, x,
                 build_self_attn(L, si, x, attn, L.attn2_in_w, L.attn2_in_b,
                                 L.attn2_out_w, L.attn2_out_b, ls,
                                 ls ? ls->val2_in : nullptr,
                                 ls ? &ls->val2_out : nullptr));
    x = ggml_add(ctx, x, build_conv_module(L.conv2, si, x, ls,
                                           ls ? ls->conv2_in : nullptr,
                                           ls ? &ls->conv2_out : nullptr));
    x = ggml_add(ctx, x, build_ff(L.ff3_in_w, L.ff3_in_b, L.ff3_out_w,
                                  L.ff3_out_b, x));
    x = bias_norm(x, L.norm_bias, L.norm_scale);
    x = bypass(orig, x, L.bypass_scale);
    return x;
  }

  // SimpleDownsample: [C, T] -> [C, ceil(T/ds)]
  struct ggml_tensor *build_downsample(struct ggml_tensor *x,
                                       struct ggml_tensor *w, int ds) {
    const int64_t C = x->ne[0], T = x->ne[1];
    const int64_t T2 = (T + ds - 1) / ds;
    const int64_t pad = T2 * ds - T;
    if (pad > 0) {
      struct ggml_tensor *last = ggml_view_2d(
          ctx, x, C, 1, x->nb[1], (size_t)(T - 1) * x->nb[1]);
      struct ggml_tensor *tmpl =
          ggml_new_tensor_2d(ctx, GGML_TYPE_F32, C, pad);
      struct ggml_tensor *rep = ggml_repeat(ctx, last, tmpl);
      x = ggml_concat(ctx, x, rep, 1);
    }
    struct ggml_tensor *x3 = ggml_reshape_3d(ctx, x, C, ds, T2);
    x3 = ggml_mul(ctx, x3, ggml_reshape_3d(ctx, w, 1, ds, 1));
    // sum over ne1: permute to [ds, C, T2], sum_rows -> [1, C, T2]
    struct ggml_tensor *p =
        ggml_cont(ctx, ggml_permute(ctx, x3, 1, 0, 2, 3)); // [ds, C, T2]
    struct ggml_tensor *sum = ggml_sum_rows(ctx, p);       // [1, C, T2]
    return ggml_reshape_2d(ctx, ggml_cont(ctx, sum), C, T2);
  }

  // SimpleUpsample + truncate: [C, T2] -> [C, T_orig]
  struct ggml_tensor *build_upsample(struct ggml_tensor *x, int ds,
                                     int64_t T_orig) {
    const int64_t C = x->ne[0], T2 = x->ne[1];
    struct ggml_tensor *x3 = ggml_reshape_3d(ctx, x, C, 1, T2);
    struct ggml_tensor *tmpl = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, C, ds, T2);
    struct ggml_tensor *rep = ggml_repeat(ctx, x3, tmpl); // [C, ds, T2]
    struct ggml_tensor *flat = ggml_reshape_2d(ctx, rep, C, ds * T2);
    if (ds * T2 == T_orig) return flat;
    return ggml_cont(ctx,
                     ggml_view_2d(ctx, flat, C, T_orig, flat->nb[1], 0));
  }

  // convert_num_channels: pad with zero channels or truncate
  struct ggml_tensor *convert_channels(struct ggml_tensor *x, int d_out) {
    const int64_t d_in = x->ne[0];
    if (d_in == d_out) return x;
    if (d_in < d_out) return ggml_pad(ctx, x, d_out - (int)d_in, 0, 0, 0);
    return ggml_cont(ctx, ggml_view_2d(ctx, x, d_out, x->ne[1], x->nb[1], 0));
  }

  // --- full offline encoder ---------------------------------------------------
  // returns enc_proj output [joiner_dim, T25]
  struct ggml_tensor *build_offline() {
    const auto &hp = m.hp_;
    fbank_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.feature_dim, T_fbank);
    ggml_set_input(fbank_in);
    ggml_set_name(fbank_in, "fbank");

    struct ggml_tensor *x = build_embed(fbank_in, false); // [dim0, T50]
    dump("embed_out", x);
    const int64_t T50 = x->ne[1];

    std::vector<struct ggml_tensor *> outputs(hp.num_stacks);
    for (int si = 0; si < hp.num_stacks; si++) {
      const auto &st = m.stacks_[si];
      const int ds = hp.downsampling[si];
      x = convert_channels(x, hp.encoder_dim[si]);
      struct ggml_tensor *sk = x; // stack input at full 50 Hz rate
      struct ggml_tensor *y = x;
      if (ds > 1) y = build_downsample(y, st.downsample_w, ds);
      const int64_t Ts = y->ne[1];
      std::vector<float> pos_host;
      xasr_compute_pos_emb((int)Ts, 0, hp.pos_dim, pos_host);
      struct ggml_tensor *pos =
          stage_input(std::move(pos_host), hp.pos_dim, 2 * Ts - 1);
      for (const auto &L : st.layers) {
        y = build_layer(L, si, y, pos, nullptr);
      }
      if (ds > 1) {
        y = build_upsample(y, ds, T50);
        y = bypass(sk, y, st.out_combiner_w);
      }
      outputs[si] = y;
      dump(("stack" + std::to_string(si) + "_out").c_str(), y);
      x = y;
    }

    // _get_full_dim_output
    struct ggml_tensor *full = outputs[hp.num_stacks - 1];
    int cur_dim = hp.encoder_dim[hp.num_stacks - 1];
    for (int i = hp.num_stacks - 2; i >= 0; i--) {
      const int d = hp.encoder_dim[i];
      if (d > cur_dim) {
        struct ggml_tensor *piece = ggml_view_2d(
            ctx, outputs[i], d - cur_dim, outputs[i]->ne[1],
            outputs[i]->nb[1], (size_t)cur_dim * ggml_element_size(outputs[i]));
        full = ggml_concat(ctx, full, ggml_cont(ctx, piece), 0);
        cur_dim = d;
      }
    }

    // downsample_output (ds=2)
    struct ggml_tensor *out =
        build_downsample(full, m.downsample_output_w, 2);
    dump("encoder_out", out);

    // joiner encoder_proj
    struct ggml_tensor *proj =
        linear(m.joiner_enc_proj_w, m.joiner_enc_proj_b, out);
    ggml_set_name(proj, "enc_proj");
    ggml_set_output(proj);
    dump("enc_proj", proj);
    return proj;
  }

  // --- streaming chunked encoder ---------------------------------------------
  // fbank chunk [80, T_in] (T_in = decode_chunk_len + pad_length). Cache and
  // mask tensors are graph inputs (uploaded / rolled by the caller); pos_emb
  // and conv edge scales are chunk-size constants staged once.
  struct ggml_tensor *build_streaming() {
    const auto &hp = m.hp_;
    fbank_in =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hp.feature_dim, T_fbank);
    ggml_set_input(fbank_in);

    // embed cache input
    const int F = (((hp.feature_dim - 1) / 2) - 1) / 2;
    const int C = m.embed_.layer3_channels;
    embed_cache_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, F, 3, C);
    ggml_set_input(embed_cache_in);

    struct ggml_tensor *x = build_embed(fbank_in, true); // [dim0, T50]
    const int64_t T50 = x->ne[1];
    const int left50 = hp.left_context_frames;

    // per-layer cache tensors, flattened over stacks
    int layer_idx = 0;
    lstream.resize(m.total_layers());
    std::vector<struct ggml_tensor *> outputs(hp.num_stacks);

    for (int si = 0; si < hp.num_stacks; si++) {
      const auto &stk = m.stacks_[si];
      const int ds = hp.downsampling[si];
      const int d = hp.encoder_dim[si];
      const int left_s = left50 / ds;
      const int64_t Ts = T50 / ds;
      const int K = hp.cnn_kernel[si];
      const int kd = hp.query_head_dim[si] * hp.num_heads[si];
      const int vd = hp.value_head_dim[si] * hp.num_heads[si];
      const int hc = 3 * d / 4;

      x = convert_channels(x, d);
      struct ggml_tensor *sk = x;
      struct ggml_tensor *y = x;
      if (ds > 1) y = build_downsample(y, stk.downsample_w, ds);

      std::vector<float> pos_host;
      xasr_compute_pos_emb((int)Ts, left_s, hp.pos_dim, pos_host);
      struct ggml_tensor *pos =
          stage_input(std::move(pos_host), hp.pos_dim, left_s + 2 * Ts - 1);

      // additive key-padding mask for the fused soft_max_ext: [k_len, Ts]
      // (soft_max_ext requires ne1 >= target length and broadcasts only over
      // heads). Uploaded per chunk until the left context fills, then it stays
      // all-zero and uploads stop.
      struct ggml_tensor *mask_t =
          ggml_new_tensor_2d(ctx, GGML_TYPE_F32, left_s + Ts, Ts);
      ggml_set_input(mask_t);
      mask_in.push_back(mask_t);
      mask_ds.push_back(ds);

      for (const auto &L : stk.layers) {
        LayerStream &ls = lstream[layer_idx++];
        ls.left = left_s;
        ls.mask = mask_t;
        ls.key_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kd, left_s);
        ls.nonlin_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, left_s);
        ls.val1_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vd, left_s);
        ls.val2_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vd, left_s);
        ls.conv1_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K / 2, d);
        ls.conv2_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K / 2, d);
        ggml_set_input(ls.key_in);
        ggml_set_input(ls.nonlin_in);
        ggml_set_input(ls.val1_in);
        ggml_set_input(ls.val2_in);
        ggml_set_input(ls.conv1_in);
        ggml_set_input(ls.conv2_in);
        y = build_layer(L, si, y, pos, &ls);
      }

      if (ds > 1) {
        y = build_upsample(y, ds, T50);
        y = bypass(sk, y, stk.out_combiner_w);
      }
      outputs[si] = y;
      x = y;
    }

    struct ggml_tensor *full = outputs[hp.num_stacks - 1];
    int cur_dim = hp.encoder_dim[hp.num_stacks - 1];
    for (int i = hp.num_stacks - 2; i >= 0; i--) {
      const int d = hp.encoder_dim[i];
      if (d > cur_dim) {
        struct ggml_tensor *piece = ggml_view_2d(
            ctx, outputs[i], d - cur_dim, outputs[i]->ne[1],
            outputs[i]->nb[1], (size_t)cur_dim * ggml_element_size(outputs[i]));
        full = ggml_concat(ctx, full, ggml_cont(ctx, piece), 0);
        cur_dim = d;
      }
    }

    struct ggml_tensor *out =
        build_downsample(full, m.downsample_output_w, 2);
    struct ggml_tensor *proj =
        linear(m.joiner_enc_proj_w, m.joiner_enc_proj_b, out);
    ggml_set_output(proj);
    return proj;
  }
};

// ---------------------------------------------------------------------------
// XASRModel::EncodeOffline
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>

#define XASR_MAX_NODES 16384

static void xasr_write_dump(const char *dir, const std::string &name,
                            struct ggml_tensor *t) {
  std::vector<float> host(ggml_nelements(t));
  ggml_backend_tensor_get(t, host.data(), 0, ggml_nbytes(t));
  std::string path = std::string(dir) + "/" + name + ".bin";
  FILE *f = fopen(path.c_str(), "wb");
  if (!f) return;
  int32_t nd = ggml_n_dims(t);
  int32_t ne[4] = {(int32_t)t->ne[0], (int32_t)t->ne[1], (int32_t)t->ne[2],
                   (int32_t)t->ne[3]};
  fwrite(&nd, sizeof(nd), 1, f);
  fwrite(ne, sizeof(ne), 1, f);
  fwrite(host.data(), sizeof(float), host.size(), f);
  fclose(f);
}

bool XASRModel::EncodeOffline(const std::vector<float> &fbank, int n_frames,
                              XASRState &st, ggml_backend_sched_t sched) {
  if (n_frames < 8) {
    RS_LOG_ERR("XASR: input too short (%d fbank frames)", n_frames);
    return false;
  }

  if (ctx_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
    gf_ = nullptr;
  }
  if (!init_compute_ctx(&ctx_, &gf_, XASR_MAX_NODES)) return false;

  XASRGraphBuilder b(*this, ctx_, n_frames);
  struct ggml_tensor *proj = b.build_offline();
  ggml_build_forward_expand(gf_, proj);
  // keep dump tensors alive in the graph
  for (auto &d : b.dumps) ggml_build_forward_expand(gf_, d.second);

  if (!ggml_backend_sched_alloc_graph(sched, gf_)) {
    RS_LOG_ERR("XASR: sched_alloc_graph failed (%d nodes)", ggml_graph_n_nodes(gf_));
    return false;
  }

  // upload inputs
  ggml_backend_tensor_set(b.fbank_in, fbank.data(), 0,
                          (size_t)n_frames * hp_.feature_dim * sizeof(float));
  for (auto &si : b.staged) {
    ggml_backend_tensor_set(si.first, si.second.data(), 0,
                            si.second.size() * sizeof(float));
  }

  if (ggml_backend_sched_graph_compute(sched, gf_) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("XASR: graph compute failed");
    return false;
  }

  const char *dump_dir = getenv("RS_XASR_DUMP_DIR");
  if (dump_dir && dump_dir[0]) {
    for (auto &d : b.dumps) xasr_write_dump(dump_dir, d.first, d.second);
  }

  // persist enc_proj to host: tensor [joiner_dim, T25] -> row-major [T25][512]
  st.n_enc_frames = (int)proj->ne[1];
  st.enc_proj.resize((size_t)st.n_enc_frames * hp_.joiner_dim);
  ggml_backend_tensor_get(proj, st.enc_proj.data(), 0,
                          st.enc_proj.size() * sizeof(float));

  ggml_free(ctx_);
  ctx_ = nullptr;
  gf_ = nullptr;
  return true;
}

// ---------------------------------------------------------------------------
// streaming: init state caches + run one chunk graph
// ---------------------------------------------------------------------------

void XASRModel::InitStreamState(XASRState &st) {
  const int left50 = hp_.left_context_frames;
  st.caches.clear();
  st.caches.resize(total_layers());
  int li = 0;
  for (int si = 0; si < hp_.num_stacks; si++) {
    const int ds = hp_.downsampling[si];
    const int d = hp_.encoder_dim[si];
    const int left_s = left50 / ds;
    const int K = hp_.cnn_kernel[si];
    const int kd = hp_.query_head_dim[si] * hp_.num_heads[si];
    const int vd = hp_.value_head_dim[si] * hp_.num_heads[si];
    const int hc = 3 * d / 4;
    for (int l = 0; l < hp_.num_layers[si]; l++, li++) {
      auto &c = st.caches[li];
      c.key.assign((size_t)kd * left_s, 0.0f);
      c.nonlin.assign((size_t)hc * left_s, 0.0f);
      c.val1.assign((size_t)vd * left_s, 0.0f);
      c.val2.assign((size_t)vd * left_s, 0.0f);
      c.conv1.assign((size_t)(K / 2) * d, 0.0f);
      c.conv2.assign((size_t)(K / 2) * d, 0.0f);
    }
  }
  const int F = (((hp_.feature_dim - 1) / 2) - 1) / 2;
  st.embed_left_pad.assign((size_t)F * 3 * embed_.layer3_channels, 0.0f);
  st.processed_lens = 0;
  st.fbank_frames_done = 0;
  st.ctx_ids.assign(hp_.context_size, hp_.blank_id);
  RunPredictor(st.ctx_ids.data(), st.dec_out);
  st.enc_proj.clear();
  st.n_enc_frames = 0;
  st.tokens.clear();
  st.stream_initialized = true;
}

bool XASRModel::EncodeStreamChunkGraph(const std::vector<float> &fbank_chunk,
                                       XASRState &st,
                                       ggml_backend_sched_t sched) {
  const int T_in = (int)(fbank_chunk.size() / hp_.feature_dim);
  const int T50 = (T_in - 7) / 2 - 3;
  ggml_backend_sched_t es = enc_sched_ ? enc_sched_ : sched;

  // Build a fresh chunk graph each call, mirroring the offline path
  // (init_compute_ctx -> build -> alloc_graph -> set inputs -> compute ->
  // read -> free). This is the pattern proven portable across CPU / Metal /
  // CUDA / Vulkan in this codebase; a persistent alloc-once graph relies on
  // backend-specific scheduler buffer semantics that differ between Metal and
  // the CPU allocator. The graph build (~1-2 ms) is negligible against the
  // ~20 ms compute. Cache state lives in host vectors (st.caches).
  ggml_backend_sched_reset(es);
  if (ctx_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
    gf_ = nullptr;
  }
  if (!init_compute_ctx(&ctx_, &gf_, XASR_MAX_NODES)) return false;

  XASRGraphBuilder b(*this, ctx_, T_in);
  struct ggml_tensor *proj = b.build_streaming();
  ggml_build_forward_expand(gf_, proj);
  for (auto &ls : b.lstream) {
    ggml_build_forward_expand(gf_, ls.key_out);
    ggml_build_forward_expand(gf_, ls.nonlin_out);
    ggml_build_forward_expand(gf_, ls.val1_out);
    ggml_build_forward_expand(gf_, ls.val2_out);
    ggml_build_forward_expand(gf_, ls.conv1_out);
    ggml_build_forward_expand(gf_, ls.conv2_out);
  }
  ggml_build_forward_expand(gf_, b.embed_cache_out);

  if (!ggml_backend_sched_alloc_graph(es, gf_)) {
    RS_LOG_ERR("XASR: streaming sched_alloc_graph failed (%d nodes)",
               ggml_graph_n_nodes(gf_));
    return false;
  }

  // chunk-size constants (pos embeddings, conv edge scales)
  for (auto &c : b.staged) {
    ggml_backend_tensor_set(c.first, c.second.data(), 0,
                            c.second.size() * sizeof(float));
  }

  // layer caches + embed left-pad from host state
  for (size_t li = 0; li < b.lstream.size(); li++) {
    auto &ls = b.lstream[li];
    auto &c = st.caches[li];
    struct ggml_tensor *ins[6] = {ls.key_in,  ls.nonlin_in, ls.val1_in,
                                  ls.val2_in, ls.conv1_in,  ls.conv2_in};
    const std::vector<float> *bufs[6] = {&c.key,  &c.nonlin, &c.val1,
                                         &c.val2, &c.conv1,  &c.conv2};
    for (int i = 0; i < 6; i++) {
      ggml_backend_tensor_set(ins[i], bufs[i]->data(), 0,
                              bufs[i]->size() * sizeof(float));
    }
  }
  ggml_backend_tensor_set(b.embed_cache_in, st.embed_left_pad.data(), 0,
                          st.embed_left_pad.size() * sizeof(float));

  // key-padding masks [k_len, Ts]: -1000 for still-empty left slots while the
  // context is cold, all-zero once processed >= left. The same [k_len] source
  // mask is replicated across Ts target rows (soft_max_ext broadcasts only
  // over heads).
  const int left50 = hp_.left_context_frames;
  std::vector<float> mask50(left50 + T50, 0.0f);
  for (int j = 0; j < left50; j++) {
    if (st.processed_lens <= (left50 - 1 - j)) mask50[j] = -1000.0f;
  }
  for (size_t si = 0; si < b.mask_in.size(); si++) {
    const int ds = b.mask_ds[si];
    struct ggml_tensor *mt = b.mask_in[si];
    const int k_len = (int)mt->ne[0];
    const int Ts = (int)mt->ne[1];
    std::vector<float> row;
    row.reserve(k_len);
    for (size_t j = 0; j < mask50.size() && (int)row.size() < k_len; j += ds)
      row.push_back(mask50[j]);
    row.resize(k_len, 0.0f);
    std::vector<float> full((size_t)k_len * Ts);
    for (int t = 0; t < Ts; t++)
      std::copy(row.begin(), row.end(), full.begin() + (size_t)t * k_len);
    ggml_backend_tensor_set(mt, full.data(), 0, full.size() * sizeof(float));
  }

  ggml_backend_tensor_set(b.fbank_in, fbank_chunk.data(), 0,
                          fbank_chunk.size() * sizeof(float));

  if (ggml_backend_sched_graph_compute(es, gf_) != GGML_STATUS_SUCCESS) {
    RS_LOG_ERR("XASR: streaming graph compute failed");
    return false;
  }

  // read the new caches back to host state for the next chunk
  for (size_t li = 0; li < b.lstream.size(); li++) {
    auto &ls = b.lstream[li];
    auto &c = st.caches[li];
    struct ggml_tensor *outs[6] = {ls.key_out,  ls.nonlin_out, ls.val1_out,
                                   ls.val2_out, ls.conv1_out,  ls.conv2_out};
    std::vector<float> *bufs[6] = {&c.key,  &c.nonlin, &c.val1,
                                   &c.val2, &c.conv1,  &c.conv2};
    for (int i = 0; i < 6; i++) {
      ggml_backend_tensor_get(outs[i], bufs[i]->data(), 0,
                              bufs[i]->size() * sizeof(float));
    }
  }
  ggml_backend_tensor_get(b.embed_cache_out, st.embed_left_pad.data(), 0,
                          st.embed_left_pad.size() * sizeof(float));

  // append enc_proj rows
  const int T25 = (int)proj->ne[1];
  const size_t old = st.enc_proj.size();
  st.enc_proj.resize(old + (size_t)T25 * hp_.joiner_dim);
  ggml_backend_tensor_get(proj, st.enc_proj.data() + old, 0,
                          (size_t)T25 * hp_.joiner_dim * sizeof(float));
  st.n_enc_frames += T25;

  ggml_free(ctx_);
  ctx_ = nullptr;
  gf_ = nullptr;

  st.processed_lens += T50;
  return true;
}
