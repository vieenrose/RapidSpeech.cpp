// Stable Audio 3 (small-music) — text-conditioned audio/music generation.
//
// Faithful ggml translation of the official MLX reference
// (optimized/mlx/models/defs/{dit_mlx,same_s_decoder}.py and sa3_pipeline.py).
// Built on RapidSpeech primitives only (no external abstraction layer).
//
// VALIDATION: this file has NOT been compiled or numerically validated here
// (the sandbox build env is unavailable). The layouts most likely to need a
// numeric cross-check against the MLX reference are flagged `// VALIDATE`.
// Per-stage dump env vars are provided:
//   RS_SA3_DUMP_LATENTS  — DiT output latents [256, T_lat]
//   RS_SA3_DUMP_DEC      — SAME-S patches [512, T_lat*16]
//
// Reference constants (small-music): DiT embed=1024 depth=20 heads=16 hd=64
// rope=32 ff=4096 mem=64 cond=768; SAME dim=768 depth=6 heads=12 hd=64 rope=32
// ff=2304 out=512 stride=16; sr=44100 stereo patch=256 (16*256=4096x).

#include "stable_audio3.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "utils/rs_log.h"

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

// =====================================================================
// File-local helpers
// =====================================================================
namespace {

enum ggml_status sa3_sched_compute(ggml_backend_sched_t sched, ggml_cgraph *gf,
                                   std::function<void(ggml_tensor *)> *cb)
{
    const bool armed = cb && (bool)*cb;
    if (armed)
    {
        auto tramp = [](ggml_tensor *t, bool ask, void *ud) -> bool {
            if (ask) return true;
            (*static_cast<std::function<void(ggml_tensor *)> *>(ud))(t);
            return true;
        };
        ggml_backend_sched_set_eval_callback(sched, tramp, cb);
    }
    enum ggml_status st = ggml_backend_sched_graph_compute(sched, gf);
    if (armed) ggml_backend_sched_set_eval_callback(sched, nullptr, nullptr);
    return st;
}

ggml_backend_t pick_backend(ggml_backend_sched_t sched)
{
    const int n = ggml_backend_sched_get_n_backends(sched);
    for (int i = 0; i < n; ++i)
    {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t d = ggml_backend_get_device(b);
        if (d && ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_CPU) return b;
    }
    return ggml_backend_sched_get_backend(sched, n - 1);
}

ggml_tensor *lin(ggml_context *c, ggml_tensor *w, ggml_tensor *b, ggml_tensor *x)
{
    ggml_tensor *y = ggml_mul_mat(c, w, x);
    if (b) y = ggml_add(c, y, b);
    return y;
}

// RMSNorm over ne0, optional weight.
ggml_tensor *rms(ggml_context *c, ggml_tensor *x, ggml_tensor *w, float eps)
{
    x = ggml_rms_norm(c, x, eps);
    if (w) x = ggml_mul(c, x, w);
    return x;
}

// DyT: gamma * tanh(alpha * x) + beta. alpha is a [1] tensor.
ggml_tensor *dyt(ggml_context *c, ggml_tensor *x, const SA3DyT &n)
{
    x = ggml_mul(c, x, n.alpha);          // broadcast scalar
    x = ggml_tanh(c, x);
    x = ggml_mul(c, x, n.gamma);
    x = ggml_add(c, x, n.beta);
    return x;
}

// GLU/SwiGLU feed-forward: proj -> split(value,gate) -> value*silu(gate) -> out.
ggml_tensor *swiglu(ggml_context *c, ggml_tensor *x, ggml_tensor *gw,
                    ggml_tensor *gb, ggml_tensor *ow, ggml_tensor *ob, int inner)
{
    ggml_tensor *h = lin(c, gw, gb, x);   // [2*inner, T, B]
    ggml_tensor *val  = ggml_view_3d(c, h, inner, h->ne[1], h->ne[2],
                                     h->nb[1], h->nb[2], 0);
    ggml_tensor *gate = ggml_view_3d(c, h, inner, h->ne[1], h->ne[2],
                                     h->nb[1], h->nb[2], (size_t)inner * h->nb[0]);
    val  = ggml_cont(c, val);
    gate = ggml_cont(c, gate);
    ggml_tensor *g = ggml_mul(c, val, ggml_silu(c, gate));
    return lin(c, ow, ob, g);
}

// Split [E,T,B] -> heads [head_dim, n_head, T, B].
ggml_tensor *to_heads(ggml_context *c, ggml_tensor *x, int hd, int nh)
{
    return ggml_reshape_4d(c, x, hd, nh, x->ne[1], x->ne[2]);
}

// RoPE (NEOX) on heads [hd,nh,T,B] with positions pos[T*B]=idx%T.
ggml_tensor *rope_heads(ggml_context *c, ggml_tensor *xh, ggml_tensor *pos,
                        int rope_dims)
{
    const int64_t hd = xh->ne[0], nh = xh->ne[1], T = xh->ne[2], B = xh->ne[3];
    ggml_tensor *r = ggml_reshape_3d(c, ggml_cont(c, xh), hd, nh, T * B);
    r = ggml_rope(c, r, pos, rope_dims, GGML_ROPE_TYPE_NEOX);
    return ggml_reshape_4d(c, r, hd, nh, T, B);
}

// Scaled dot-product attention. q:[hd,nh,Tq,B] k,v:[hd,nh,Tk,B] -> [E,Tq,B].
ggml_tensor *sdpa(ggml_context *c, ggml_tensor *q, ggml_tensor *k,
                  ggml_tensor *v, float scale)
{
    const int64_t hd = q->ne[0], nh = q->ne[1], Tq = q->ne[2], B = q->ne[3];
    ggml_tensor *qp = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3)); // [hd,Tq,nh,B]
    ggml_tensor *kp = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3)); // [hd,Tk,nh,B]
    ggml_tensor *s = ggml_mul_mat(c, kp, qp);                       // [Tk,Tq,nh,B]
    s = ggml_scale(c, s, scale);
    s = ggml_soft_max(c, s);
    ggml_tensor *vp = ggml_cont(c, ggml_permute(c,
                          ggml_permute(c, v, 0, 2, 1, 3), 1, 0, 2, 3)); // [Tk,hd,nh,B]
    ggml_tensor *o = ggml_mul_mat(c, vp, s);                        // [hd,Tq,nh,B]
    o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));              // [hd,nh,Tq,B]
    return ggml_reshape_3d(c, o, hd * nh, Tq, B);
}

// Conv1d kernel=1 (== channel-wise linear) with residual. w ggml ne=[1,Cin,Cout].
ggml_tensor *conv1x1_res(ggml_context *c, ggml_tensor *w, ggml_tensor *x)
{
    ggml_tensor *w2 = ggml_reshape_2d(c, w, w->ne[1], w->ne[2]); // [Cin,Cout]
    ggml_tensor *y = ggml_mul_mat(c, w2, x);                      // [Cout,T,B]
    return ggml_add(c, y, x);
}

// Conv1d kernel=3 pad=1 'same'. x:[C_in,T,B], w ggml ne=[3,C_in,C_out]. -> [C_out,T,B].
ggml_tensor *conv1d_k3p1(ggml_context *c, ggml_tensor *w, ggml_tensor *b,
                         ggml_tensor *x)
{
    ggml_tensor *h = ggml_cont(c, ggml_permute(c, x, 1, 0, 2, 3)); // [T,C_in,B]
    h = ggml_pad_ext(c, h, 1, 1, 0, 0, 0, 0, 0, 0);               // pad T by 1 each side
    ggml_tensor *im = ggml_im2col(c, w, h, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
    ggml_tensor *wm = ggml_reshape_2d(c, w, w->ne[0] * w->ne[1], w->ne[2]);
    if (im->ne[2] > 1)
        wm = ggml_repeat_4d(c, wm, wm->ne[0], wm->ne[1], im->ne[2], wm->ne[3]);
    ggml_tensor *y = ggml_mul_mat(c, im, wm);                     // [T,C_out,B]
    if (b)
    {
        ggml_tensor *bb = ggml_reshape_3d(c, b, 1, b->ne[0], 1);
        y = ggml_add(c, y, bb);
    }
    return ggml_cont(c, ggml_permute(c, y, 1, 0, 2, 3));          // [C_out,T,B]
}

ggml_tensor *unsq1(ggml_context *c, ggml_tensor *x)   // [E,B] -> [E,1,B]
{
    return ggml_view_4d(c, x, x->ne[0], 1, x->ne[1], x->ne[2],
                        x->nb[1], x->nb[1], x->nb[2], 0);
}

} // namespace

// =====================================================================
// Construction / state
// =====================================================================

StableAudio3Model::StableAudio3Model()
{
    meta_.arch_name             = "stable-audio3";
    meta_.audio_sample_rate     = hparams_.audio_sample_rate;
    meta_.use_external_frontend = true;
}

StableAudio3Model::~StableAudio3Model()
{
    if (tables_buf_) ggml_backend_buffer_free(tables_buf_);
    if (tables_ctx_) ggml_free(tables_ctx_);
}

std::shared_ptr<RSState> StableAudio3Model::CreateState()
{
    auto st = std::make_shared<StableAudio3State>();
    st->duration_s = pending_duration_s_;
    st->rng_.seed((uint32_t)seed_);
    return st;
}

// =====================================================================
// Loading
// =====================================================================

bool StableAudio3Model::LoadHParams(gguf_context *g)
{
    auto i32 = [&](const std::string &k, int d) {
        int id = gguf_find_key(g, k.c_str());
        return id >= 0 ? (int)gguf_get_val_i32(g, id) : d;
    };
    auto f32 = [&](const std::string &k, float d) {
        int id = gguf_find_key(g, k.c_str());
        return id >= 0 ? gguf_get_val_f32(g, id) : d;
    };
    const std::string A = "stable-audio3.";
    hparams_.latent_dim       = i32(A + "latent_dim", hparams_.latent_dim);
    hparams_.audio_sample_rate= i32(A + "audio_sample_rate", hparams_.audio_sample_rate);
    hparams_.audio_channels   = i32(A + "audio_channels", hparams_.audio_channels);
    hparams_.patch_size       = i32(A + "patch_size", hparams_.patch_size);
    hparams_.same_downsample  = i32(A + "same_downsample", hparams_.same_downsample);
    hparams_.dit_embed_dim    = i32(A + "dit_embed_dim", hparams_.dit_embed_dim);
    hparams_.dit_depth        = i32(A + "dit_depth", hparams_.dit_depth);
    hparams_.dit_n_head       = i32(A + "dit_n_head", hparams_.dit_n_head);
    hparams_.dit_head_dim     = i32(A + "dit_head_dim", hparams_.dit_head_dim);
    hparams_.dit_rope_dims    = i32(A + "dit_rope_dims", hparams_.dit_rope_dims);
    hparams_.dit_ff_inner     = i32(A + "dit_ff_inner", hparams_.dit_ff_inner);
    hparams_.dit_n_memory     = i32(A + "dit_n_memory", hparams_.dit_n_memory);
    hparams_.cond_token_dim   = i32(A + "cond_token_dim", hparams_.cond_token_dim);
    hparams_.same_dim         = i32(A + "same_dim", hparams_.same_dim);
    hparams_.same_depth       = i32(A + "same_depth", hparams_.same_depth);
    hparams_.same_n_head      = i32(A + "same_n_head", hparams_.same_n_head);
    hparams_.same_head_dim    = i32(A + "same_head_dim", hparams_.same_head_dim);
    hparams_.same_rope_dims   = i32(A + "same_rope_dims", hparams_.same_rope_dims);
    hparams_.same_ff_inner    = i32(A + "same_ff_inner", hparams_.same_ff_inner);
    hparams_.same_out_ch      = i32(A + "same_out_ch", hparams_.same_out_ch);
    hparams_.same_stride      = i32(A + "same_stride", hparams_.same_stride);
    hparams_.n_sample_steps   = i32(A + "n_sample_steps", hparams_.n_sample_steps);
    hparams_.cfg_scale        = f32(A + "cfg_scale", hparams_.cfg_scale);
    RS_LOG_INFO("[SA3] DiT(E=%d L=%d H=%d) SAME(d=%d L=%d) sr=%d steps=%d\n",
                hparams_.dit_embed_dim, hparams_.dit_depth, hparams_.dit_n_head,
                hparams_.same_dim, hparams_.same_depth, hparams_.audio_sample_rate,
                hparams_.n_sample_steps);
    return true;
}

bool StableAudio3Model::AllocTables(ggml_backend_t backend)
{
    const int half = hparams_.timestep_feat / 2;     // 128
    std::vector<float> freqs(half);
    const float lo = std::log(0.5f), hi = std::log(10000.0f);
    for (int i = 0; i < half; ++i)
    {
        float ramp = (half > 1) ? (float)i / (half - 1) : 0.f;
        freqs[i] = std::exp(ramp * (hi - lo) + lo) * 2.0f * (float)M_PI;
    }
    ggml_init_params ip = { 2 * ggml_tensor_overhead() + 512, nullptr, true };
    tables_ctx_ = ggml_init(ip);
    timestep_freqs_ = ggml_new_tensor_1d(tables_ctx_, GGML_TYPE_F32, half);
    ggml_set_name(timestep_freqs_, "sa3.tstep_freqs");
    tables_buf_ = ggml_backend_alloc_ctx_tensors(tables_ctx_, backend);
    if (!tables_buf_) { ggml_free(tables_ctx_); tables_ctx_ = nullptr; return false; }
    ggml_backend_tensor_set(timestep_freqs_, freqs.data(), 0, half * sizeof(float));
    seconds_freqs_ = timestep_freqs_;   // identical formula
    return true;
}

void StableAudio3Model::BuildSchedule()
{
    // build_pingpong_schedule: linspace(1,0,N+1) warped by logsnr_shift, t[0]=1.
    const int N = hparams_.n_sample_steps;
    t_schedule_.resize(N + 1);
    const float a = hparams_.logsnr_anchor, e = hparams_.logsnr_end;
    for (int i = 0; i <= N; ++i)
    {
        float t = 1.0f - (float)i / (float)N;           // linspace(1,0)
        float logsnr = e - t * (e - a);
        float warped = 1.0f / (1.0f + std::exp(logsnr)); // sigmoid(-logsnr)
        if (t <= 0.f) warped = 0.f;
        if (t >= 1.f) warped = 1.f;
        t_schedule_[i] = warped;
    }
    t_schedule_[0] = 1.0f;   // re-anchor start
}

// ---- SentencePiece-unigram tokenizer (mirrors indextts2_tokenizer.cpp) ----
bool SA3T5Tokenizer::load(std::vector<std::string> toks, std::vector<float> sc)
{
    id_to_token = std::move(toks);
    scores = std::move(sc);
    token_to_id.clear();
    token_to_id.reserve(id_to_token.size());
    for (size_t i = 0; i < id_to_token.size(); ++i) token_to_id[id_to_token[i]] = (int32_t)i;
    loaded = !id_to_token.empty();
    return loaded;
}

std::vector<int32_t> SA3T5Tokenizer::encode(const std::string &text) const
{
    // SentencePiece: leading ▁ + ' ' -> ▁ (U+2581). Viterbi over unigram scores.
    if (!loaded) return {};
    std::string proc = "\xe2\x96\x81";
    for (char c : text) proc += (c == ' ') ? std::string("\xe2\x96\x81") : std::string(1, c);
    const int N = (int)proc.size();
    const int MAXP = 48;
    std::vector<float> best(N + 1, -1e30f);
    std::vector<int> blen(N + 1, 0);
    best[0] = 0.f;
    for (int i = 0; i < N; ++i)
    {
        if (best[i] <= -1e29f) continue;
        for (int len = 1; len <= std::min(MAXP, N - i); ++len)
        {
            auto it = token_to_id.find(proc.substr(i, len));
            if (it != token_to_id.end())
            {
                float sc = (it->second < (int32_t)scores.size()) ? scores[it->second] : -20.f;
                float tot = best[i] + sc;
                if (tot > best[i + len]) { best[i + len] = tot; blen[i + len] = len; }
            }
        }
        if (best[i + 1] <= -1e29f) { best[i + 1] = best[i] - 20.f; blen[i + 1] = 1; }
    }
    std::vector<std::pair<int,int>> seg;
    for (int pos = N; pos > 0; ) { int len = blen[pos]; seg.push_back({pos-len,len}); pos -= len; }
    std::reverse(seg.begin(), seg.end());
    std::vector<int32_t> out;
    for (auto &s : seg)
    {
        auto it = token_to_id.find(proc.substr(s.first, s.second));
        out.push_back(it != token_to_id.end() ? it->second : 2 /*<unk>*/);
    }
    return out;
}

bool StableAudio3Model::LoadT5Tokenizer(gguf_context *g)
{
    int kt = gguf_find_key(g, "tokenizer.ggml.tokens");
    int ks = gguf_find_key(g, "tokenizer.ggml.scores");
    if (kt < 0 || ks < 0) { RS_LOG_WARN("[SA3] no T5 tokenizer in GGUF\n"); return false; }
    const int n = (int)gguf_get_arr_n(g, kt);
    std::vector<std::string> toks(n);
    for (int i = 0; i < n; ++i) toks[i] = gguf_get_arr_str(g, kt, i);
    std::vector<float> sc(n, 0.f);
    const float *sd = (const float *)gguf_get_arr_data(g, ks);
    if (sd) std::memcpy(sc.data(), sd, (size_t)n * sizeof(float));
    return t5_tok_.load(std::move(toks), std::move(sc));
}

bool StableAudio3Model::LoadTextEncoder(const std::unique_ptr<rs_context_t> &ctx,
                                        std::map<std::string, ggml_tensor *> &t)
{
    auto get = [&](const std::string &n, ggml_tensor **d) {
        auto it = t.find(n);
        if (it == t.end()) { RS_LOG_WARN("[SA3] missing %s\n", n.c_str()); return false; }
        *d = it->second; return true;
    };
    bool ok = true;
    ok &= get("txt.embed_tokens.weight", &t5_embed_);
    ok &= get("txt.norm.weight", &t5_norm_);
    t5_layers_.resize(hparams_.txt_layers);
    char b[256];
    for (int i = 0; i < hparams_.txt_layers; ++i)
    {
        auto &L = t5_layers_[i];
        auto f = [&](const char *s){ snprintf(b,sizeof(b),"txt.layers.%d.%s",i,s); return std::string(b); };
        ok &= get(f("self_attn.q_proj.weight"), &L.q_proj);
        ok &= get(f("self_attn.k_proj.weight"), &L.k_proj);
        ok &= get(f("self_attn.v_proj.weight"), &L.v_proj);
        ok &= get(f("self_attn.o_proj.weight"), &L.o_proj);
        ok &= get(f("mlp.gate_proj.weight"), &L.gate_proj);
        ok &= get(f("mlp.up_proj.weight"),   &L.up_proj);
        ok &= get(f("mlp.down_proj.weight"), &L.down_proj);
        ok &= get(f("pre_self_attn_layernorm.weight"),  &L.n_pre_attn);
        ok &= get(f("post_self_attn_layernorm.weight"), &L.n_post_attn);
        ok &= get(f("pre_feedforward_layernorm.weight"),  &L.n_pre_ff);
        ok &= get(f("post_feedforward_layernorm.weight"), &L.n_post_ff);
    }
    bool tok = ctx && ctx->ctx_gguf && LoadT5Tokenizer(ctx->ctx_gguf);
    t5_loaded_ = ok && tok;
    if (!t5_loaded_) RS_LOG_WARN("[SA3] T5Gemma not fully loaded — DiT runs unconditional\n");
    return true;   // optional: allow unconditional bring-up if T5 absent
}

bool StableAudio3Model::LoadConditioner(std::map<std::string, ggml_tensor *> &t)
{
    auto get = [&](const std::string &n, ggml_tensor **d) {
        auto it = t.find(n); if (it == t.end()) return false; *d = it->second; return true;
    };
    get("cond.seconds_total.weight", &seconds_w_);
    get("cond.seconds_total.bias",   &seconds_b_);
    get("cond.prompt.padding_embedding", &prompt_pad_);
    return true;   // optional during bring-up
}

bool StableAudio3Model::LoadDiT(std::map<std::string, ggml_tensor *> &t)
{
    auto get = [&](const std::string &n, ggml_tensor **d, bool req = true) {
        auto it = t.find(n);
        if (it == t.end()) { if (req) RS_LOG_ERR("[SA3] missing %s\n", n.c_str()); return !req; }
        *d = it->second; return true;
    };
    bool ok = true;
    ok &= get("dit.preprocess_conv.weight", &pre_conv_);
    ok &= get("dit.postprocess_conv.weight", &post_conv_);
    ok &= get("dit.to_cond_embed.0.weight", &to_cond0_);
    ok &= get("dit.to_cond_embed.2.weight", &to_cond2_);
    ok &= get("dit.to_global_embed.0.weight", &to_global0_);
    ok &= get("dit.to_global_embed.2.weight", &to_global2_);
    ok &= get("dit.to_timestep_embed.0.weight", &to_tstep0_w_);
    ok &= get("dit.to_timestep_embed.0.bias",   &to_tstep0_b_);
    ok &= get("dit.to_timestep_embed.2.weight", &to_tstep2_w_);
    ok &= get("dit.to_timestep_embed.2.bias",   &to_tstep2_b_);
    ok &= get("dit.transformer.project_in.weight",  &project_in_);
    ok &= get("dit.transformer.project_out.weight", &project_out_);
    ok &= get("dit.transformer.memory_tokens", &memory_tokens_);
    ok &= get("dit.transformer.global_cond_embedder.0.weight", &gce0_w_);
    ok &= get("dit.transformer.global_cond_embedder.0.bias",   &gce0_b_);
    ok &= get("dit.transformer.global_cond_embedder.2.weight", &gce2_w_);
    ok &= get("dit.transformer.global_cond_embedder.2.bias",   &gce2_b_);

    dit_blocks_.resize(hparams_.dit_depth);
    char b[256];
    for (int i = 0; i < hparams_.dit_depth; ++i)
    {
        auto &L = dit_blocks_[i];
        auto f = [&](const char *s){ snprintf(b,sizeof(b),"dit.transformer.layers.%d.%s",i,s); return std::string(b); };
        ok &= get(f("pre_norm.weight"), &L.pre_norm);
        ok &= get(f("self_attn.to_qkv.weight"), &L.self_qkv);
        ok &= get(f("self_attn.to_out.weight"), &L.self_out);
        ok &= get(f("self_attn.q_norm.weight"), &L.self_q_norm);
        ok &= get(f("self_attn.k_norm.weight"), &L.self_k_norm);
        ok &= get(f("cross_attend_norm.weight"), &L.cross_norm);
        ok &= get(f("cross_attn.to_q.weight"),  &L.cross_q);
        ok &= get(f("cross_attn.to_kv.weight"), &L.cross_kv);
        ok &= get(f("cross_attn.to_out.weight"),&L.cross_out);
        ok &= get(f("cross_attn.q_norm.weight"),&L.cross_q_norm);
        ok &= get(f("cross_attn.k_norm.weight"),&L.cross_k_norm);
        ok &= get(f("ff_norm.weight"), &L.ff_norm);
        ok &= get(f("ff.ff.0.proj.weight"), &L.ff_glu_w);
        ok &= get(f("ff.ff.0.proj.bias"),   &L.ff_glu_b);
        ok &= get(f("ff.ff.2.weight"), &L.ff_out_w);
        ok &= get(f("ff.ff.2.bias"),   &L.ff_out_b);
        ok &= get(f("to_scale_shift_gate"), &L.scale_shift_gate);
        ok &= get(f("to_local_embed.seq.0.weight"), &L.local0_w);
        ok &= get(f("to_local_embed.seq.0.bias"),   &L.local0_b);
        ok &= get(f("to_local_embed.seq.2.weight"), &L.local2_w);
        ok &= get(f("to_local_embed.seq.2.bias"),   &L.local2_b);
    }
    return ok;
}

bool StableAudio3Model::LoadSameDecoder(std::map<std::string, ggml_tensor *> &t)
{
    auto get = [&](const std::string &n, ggml_tensor **d, bool req = true) {
        auto it = t.find(n);
        if (it == t.end()) { if (req) RS_LOG_ERR("[SA3] missing %s\n", n.c_str()); return !req; }
        *d = it->second; return true;
    };
    bool ok = true;
    ok &= get("same.running_std", &same_running_std_);
    ok &= get("same.project_in.weight", &same_proj_in_w_);
    ok &= get("same.project_in.bias",   &same_proj_in_b_);
    ok &= get("same.new_tokens", &same_new_tokens_);
    ok &= get("same.mapping.weight", &same_map_w_);
    ok &= get("same.mapping.bias",   &same_map_b_, false);

    same_blocks_.resize(hparams_.same_depth);
    char b[256];
    for (int i = 0; i < hparams_.same_depth; ++i)
    {
        auto &L = same_blocks_[i];
        auto f = [&](const char *s){ snprintf(b,sizeof(b),"same.blocks.%d.%s",i,s); return std::string(b); };
        ok &= get(f("pre_norm.alpha"), &L.pre_norm.alpha);
        ok &= get(f("pre_norm.gamma"), &L.pre_norm.gamma);
        ok &= get(f("pre_norm.beta"),  &L.pre_norm.beta);
        ok &= get(f("attn.to_qkv.weight"), &L.to_qkv);
        ok &= get(f("attn.to_out.weight"), &L.to_out);
        ok &= get(f("attn.q_norm.alpha"), &L.q_norm.alpha);
        ok &= get(f("attn.q_norm.gamma"), &L.q_norm.gamma);
        ok &= get(f("attn.q_norm.beta"),  &L.q_norm.beta);
        ok &= get(f("attn.k_norm.alpha"), &L.k_norm.alpha);
        ok &= get(f("attn.k_norm.gamma"), &L.k_norm.gamma);
        ok &= get(f("attn.k_norm.beta"),  &L.k_norm.beta);
        ok &= get(f("ff_norm.alpha"), &L.ff_norm.alpha);
        ok &= get(f("ff_norm.gamma"), &L.ff_norm.gamma);
        ok &= get(f("ff_norm.beta"),  &L.ff_norm.beta);
        ok &= get(f("ff.glu_proj.weight"), &L.ff_glu_w);
        ok &= get(f("ff.glu_proj.bias"),   &L.ff_glu_b);
        ok &= get(f("ff.proj_out.weight"), &L.ff_out_w);
        ok &= get(f("ff.proj_out.bias"),   &L.ff_out_b);
    }
    same_loaded_ = ok;
    return ok;
}

bool StableAudio3Model::Load(const std::unique_ptr<rs_context_t> &ctx,
                             ggml_backend_t backend)
{
    backend_   = backend;
    gguf_data_ = ctx ? ctx->gguf_data : nullptr;
    if (!ctx || !ctx->ctx_gguf) { RS_LOG_ERR("[SA3] missing GGUF\n"); return false; }
    if (!LoadHParams(ctx->ctx_gguf)) return false;
    meta_.audio_sample_rate = hparams_.audio_sample_rate;

    std::map<std::string, ggml_tensor *> t;
    if (gguf_data_)
        for (ggml_tensor *x = ggml_get_first_tensor(gguf_data_); x;
             x = ggml_get_next_tensor(gguf_data_, x))
            t[ggml_get_name(x)] = x;

    if (!LoadTextEncoder(ctx, t)) return false;
    if (!LoadConditioner(t)) return false;
    if (!LoadDiT(t)) return false;
    if (!LoadSameDecoder(t)) return false;
    if (!AllocTables(backend)) return false;
    BuildSchedule();
    RS_LOG_INFO("[SA3] loaded — %zu DiT blocks, %zu SAME blocks\n",
                dit_blocks_.size(), same_blocks_.size());
    return true;
}

// =====================================================================
// Entry points
// =====================================================================

bool StableAudio3Model::PushText(RSState &s, const char *text,
                                 const char *lang, const char *instruct)
{
    (void)lang; (void)instruct;
    auto &st = static_cast<StableAudio3State &>(s);
    st.prompt = text ? text : "";
    st.duration_s = pending_duration_s_;
    return true;
}

bool StableAudio3Model::Encode(const std::vector<float> &in, RSState &s,
                               ggml_backend_sched_t sched)
{
    (void)in;
    return RunTextEncoder(static_cast<StableAudio3State &>(s), sched);
}

bool StableAudio3Model::Decode(RSState &s, ggml_backend_sched_t sched)
{
    auto &st = static_cast<StableAudio3State &>(s);
    if (!RunDiffusion(st, sched)) return false;
    if (!RunSameDecoder(st, sched)) return false;
    st.done = true;
    return true;
}

int StableAudio3Model::GetAudioOutput(RSState &s, float **out)
{
    auto &st = static_cast<StableAudio3State &>(s);
    if (st.audio_output.empty()) { if (out) *out = nullptr; return 0; }
    if (out) *out = st.audio_output.data();
    return (int)st.audio_output.size();
}

// T5Gemma encoder graph. ids:[S] (I32, single batch), add_mask:[S] additive.
// Returns last_hidden_state [hidden, S].
ggml_tensor *StableAudio3Model::BuildT5Encoder(ggml_context *c, ggml_tensor *ids,
                                               ggml_tensor *add_mask, int64_t S) const
{
    const int H = hparams_.txt_hidden, nh = hparams_.txt_heads, hd = hparams_.txt_head_dim;
    const float eps = hparams_.txt_rms_eps, softcap = hparams_.txt_softcap;
    const float scaling = 1.0f / std::sqrt((float)hparams_.txt_q_scalar);

    // Gemma RMSNorm: rms_norm(x) * (1 + w).
    auto gnorm = [&](ggml_tensor *x, ggml_tensor *w) {
        ggml_tensor *n = ggml_rms_norm(c, x, eps);
        ggml_tensor *wp = ggml_scale_bias(c, w, 1.f, 1.f);   // w + 1
        return ggml_mul(c, n, wp);
    };

    // embed * sqrt(H)
    ggml_tensor *x = ggml_get_rows(c, t5_embed_, ids);        // [H, S]
    x = ggml_scale(c, x, std::sqrt((float)H));

    // positions 0..S-1 for RoPE (I32 input, filled host-side by RunTextEncoder)
    ggml_tensor *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, S);
    ggml_set_input(pos);
    ggml_set_name(pos, "t5.pos");

    for (int i = 0; i < hparams_.txt_layers; ++i)
    {
        const auto &L = t5_layers_[i];
        ggml_tensor *h = gnorm(x, L.n_pre_attn);
        ggml_tensor *q = ggml_reshape_3d(c, ggml_mul_mat(c, L.q_proj, h), hd, nh, S);
        ggml_tensor *k = ggml_reshape_3d(c, ggml_mul_mat(c, L.k_proj, h), hd, nh, S);
        ggml_tensor *v = ggml_reshape_3d(c, ggml_mul_mat(c, L.v_proj, h), hd, nh, S);
        q = ggml_rope(c, q, pos, hd, GGML_ROPE_TYPE_NEOX);
        k = ggml_rope(c, k, pos, hd, GGML_ROPE_TYPE_NEOX);
        // [hd,nh,S] -> [hd,S,nh]
        q = ggml_cont(c, ggml_permute(c, q, 0, 2, 1, 3));
        k = ggml_cont(c, ggml_permute(c, k, 0, 2, 1, 3));
        ggml_tensor *qk = ggml_mul_mat(c, k, q);              // [Sk, Sq, nh]
        qk = ggml_scale(c, qk, scaling);
        // softcap: tanh(qk/cap)*cap
        qk = ggml_scale(c, ggml_tanh(c, ggml_scale(c, qk, 1.f/softcap)), softcap);
        if (add_mask) qk = ggml_add(c, qk, add_mask);         // [Sk,1,1] broadcast
        qk = ggml_soft_max(c, qk);
        // v:[hd,nh,S] -> [S,hd,nh] (matches sdpa helper layout)
        ggml_tensor *vp = ggml_cont(c, ggml_permute(c, ggml_permute(c, v, 0, 2, 1, 3), 1, 0, 2, 3));
        ggml_tensor *o = ggml_mul_mat(c, vp, qk);             // [hd,Sq,nh]
        o = ggml_cont(c, ggml_permute(c, o, 0, 2, 1, 3));     // [hd,nh,Sq]
        o = ggml_reshape_2d(c, o, hd * nh, S);
        o = ggml_mul_mat(c, L.o_proj, o);
        o = gnorm(o, L.n_post_attn);
        x = ggml_add(c, x, o);

        ggml_tensor *hf = gnorm(x, L.n_pre_ff);
        ggml_tensor *gate = ggml_gelu(c, ggml_mul_mat(c, L.gate_proj, hf));   // gelu_approx
        ggml_tensor *up = ggml_mul_mat(c, L.up_proj, hf);
        ggml_tensor *ff = ggml_mul_mat(c, L.down_proj, ggml_mul(c, gate, up));
        ff = gnorm(ff, L.n_post_ff);
        x = ggml_add(c, x, ff);
    }
    return gnorm(x, t5_norm_);   // [H, S]
}

bool StableAudio3Model::RunTextEncoder(StableAudio3State &s, ggml_backend_sched_t sched)
{
    s.text_cond.clear();
    s.n_text_tokens = 0;
    if (!t5_loaded_ || s.prompt.empty()) return true;   // -> unconditional

    std::vector<int32_t> ids = t5_tok_.encode(s.prompt);
    const int max_len = hparams_.txt_max_len;
    if ((int)ids.size() > max_len) ids.resize(max_len);
    const int S = (int)ids.size();
    if (S == 0) return true;
    const int H = hparams_.txt_hidden;

    ggml_backend_t be = pick_backend(sched);
    const size_t CTX = (size_t)SA3_MAX_NODES * (sizeof(ggml_tensor) + 320);
    ggml_init_params ip = { CTX, nullptr, true };
    ggml_context *c = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(c, SA3_MAX_NODES, false);

    ggml_tensor *ids_t = ggml_new_tensor_1d(c, GGML_TYPE_I32, S);
    ggml_set_input(ids_t); ggml_set_name(ids_t, "t5.ids");
    // No padding within the real-token window (we trim to S), so add_mask=null.
    ggml_tensor *hs = BuildT5Encoder(c, ids_t, nullptr, S);
    ggml_build_forward_expand(gf, hs);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) { RS_LOG_ERR("[SA3] t5 alloc\n"); ggml_free(c); return false; }
    ggml_backend_tensor_set(ids_t, ids.data(), 0, ids.size()*sizeof(int32_t));
    for (ggml_tensor *t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t))
        if (std::strcmp(ggml_get_name(t), "t5.pos") == 0 && t->buffer)
        {
            std::vector<int32_t> p(S); for (int j = 0; j < S; ++j) p[j] = j;
            ggml_backend_tensor_set(t, p.data(), 0, (size_t)S*sizeof(int32_t));
        }
    (void)be;
    if (sa3_sched_compute(sched, gf, &imatrix_cb_) != GGML_STATUS_SUCCESS) { RS_LOG_ERR("[SA3] t5 compute\n"); ggml_free(c); return false; }

    s.text_cond.assign((size_t)H * S, 0.f);
    ggml_backend_tensor_get(hs, s.text_cond.data(), 0, s.text_cond.size()*sizeof(float));
    s.n_text_tokens = S;
    ggml_free(c); ggml_backend_sched_reset(sched);
    RS_LOG_INFO("[SA3] T5Gemma encoded %d tokens\n", S);
    return true;
}

// =====================================================================
// DiT block (ContinuousTransformer layer). x:[E,T,B], context:[E,Tc,B],
// g6:[6E,B] (= scale_shift_gate + global_cond_embedder), pos:[T*B].
// =====================================================================

ggml_tensor *StableAudio3Model::BuildDiTBlock(ggml_context *c, const SA3DiTBlock &L,
                                              ggml_tensor *x, ggml_tensor *context,
                                              ggml_tensor *g6, ggml_tensor *pos) const
{
    const int E = hparams_.dit_embed_dim, nh = hparams_.dit_n_head;
    const int hd = hparams_.dit_head_dim, rd = hparams_.dit_rope_dims;
    const float qeps = hparams_.qk_rms_eps, neps = hparams_.rms_eps;
    const float scale = 1.0f / std::sqrt((float)hd);
    const int64_t T = x->ne[1];

    auto split6 = [&](int i){ return ggml_view_2d(c, g6, E, g6->ne[1], g6->nb[1], (size_t)i*E*g6->nb[0]); };
    ggml_tensor *sc_s = split6(0), *sh_s = split6(1), *g_s = split6(2);
    ggml_tensor *sc_f = split6(3), *sh_f = split6(4), *g_f = split6(5);

    // ----- self-attn -----
    ggml_tensor *h = rms(c, x, L.pre_norm, neps);
    ggml_tensor *one_sc = ggml_scale_bias(c, ggml_cont(c, sc_s), 1.f, 1.f);
    h = ggml_mul(c, h, unsq1(c, one_sc));
    h = ggml_add(c, h, unsq1(c, ggml_cont(c, sh_s)));

    ggml_tensor *qkv = ggml_mul_mat(c, L.self_qkv, h);     // [3E,T,B]
    auto part = [&](int i){ return ggml_cont(c, ggml_view_3d(c, qkv, E, T, h->ne[2], qkv->nb[1], qkv->nb[2], (size_t)i*E*qkv->nb[0])); };
    ggml_tensor *q = to_heads(c, part(0), hd, nh);
    ggml_tensor *k = to_heads(c, part(1), hd, nh);
    ggml_tensor *v = to_heads(c, part(2), hd, nh);
    q = rms(c, q, L.self_q_norm, qeps);
    k = rms(c, k, L.self_k_norm, qeps);
    q = rope_heads(c, q, pos, rd);
    k = rope_heads(c, k, pos, rd);
    ggml_tensor *a = sdpa(c, q, k, v, scale);
    a = ggml_mul_mat(c, L.self_out, a);
    // gate: sigmoid(1 - gate_self)
    ggml_tensor *gg = ggml_sigmoid(c, ggml_scale_bias(c, ggml_cont(c, g_s), -1.f, 1.f));
    a = ggml_mul(c, a, unsq1(c, gg));
    x = ggml_add(c, x, a);

    // ----- cross-attn (no rope, ungated residual) -----
    {
        ggml_tensor *xc = rms(c, x, L.cross_norm, neps);
        ggml_tensor *cq = to_heads(c, ggml_mul_mat(c, L.cross_q, xc), hd, nh);
        ggml_tensor *kv = ggml_mul_mat(c, L.cross_kv, context);   // [2E,Tc,B]
        const int64_t Tc = context->ne[1];
        auto cpart = [&](int i){ return ggml_cont(c, ggml_view_3d(c, kv, E, Tc, kv->ne[2], kv->nb[1], kv->nb[2], (size_t)i*E*kv->nb[0])); };
        ggml_tensor *ck = to_heads(c, cpart(0), hd, nh);
        ggml_tensor *cv = to_heads(c, cpart(1), hd, nh);
        cq = rms(c, cq, L.cross_q_norm, qeps);
        ck = rms(c, ck, L.cross_k_norm, qeps);
        ggml_tensor *co = sdpa(c, cq, ck, cv, scale);
        co = ggml_mul_mat(c, L.cross_out, co);
        x = ggml_add(c, x, co);
    }

    // ----- local cond add (text-to-audio: local_add_cond = zeros) -----
    // With a zero input, to_local_embed(0) reduces to a per-channel constant:
    //   le = local2_w @ silu(local0_b) + local2_b   (shape [E])
    // It is added to the latent positions only (memory-token prefix gets 0 in
    // the reference, where local_emb is concatenated with zeros for memory).
    {
        const int64_t Bp = x->ne[2];
        const int mem = hparams_.dit_n_memory;
        ggml_tensor *s0 = ggml_silu(c, L.local0_b);          // [E]
        ggml_tensor *le = ggml_add(c, ggml_mul_mat(c, L.local2_w, s0), L.local2_b); // [E]
        ggml_tensor *x_body = ggml_cont(c, ggml_view_3d(c, x, x->ne[0], T - mem, Bp,
                                          x->nb[1], x->nb[2], (size_t)mem*x->nb[1]));
        ggml_tensor *x_mem = ggml_cont(c, ggml_view_3d(c, x, x->ne[0], mem, Bp,
                                          x->nb[1], x->nb[2], 0));
        x_body = ggml_add(c, x_body, le);                    // broadcast le over T,B
        x = ggml_concat(c, x_mem, x_body, 1);
    }

    // ----- ff -----
    ggml_tensor *hf = rms(c, x, L.ff_norm, neps);
    ggml_tensor *one_scf = ggml_scale_bias(c, ggml_cont(c, sc_f), 1.f, 1.f);
    hf = ggml_mul(c, hf, unsq1(c, one_scf));
    hf = ggml_add(c, hf, unsq1(c, ggml_cont(c, sh_f)));
    hf = swiglu(c, hf, L.ff_glu_w, L.ff_glu_b, L.ff_out_w, L.ff_out_b, hparams_.dit_ff_inner);
    ggml_tensor *ggf = ggml_sigmoid(c, ggml_scale_bias(c, ggml_cont(c, g_f), -1.f, 1.f));
    hf = ggml_mul(c, hf, unsq1(c, ggf));
    x = ggml_add(c, x, hf);
    return x;
}

// DiT forward: returns velocity v [256, T_lat, B]. t baked as scalar.
ggml_tensor *StableAudio3Model::BuildDiTForward(ggml_context *c, ggml_tensor *x,
                                                float t_value, ggml_tensor *cross_raw,
                                                ggml_tensor *global_raw,
                                                ggml_tensor *&pos_out) const
{
    const int E = hparams_.dit_embed_dim;
    const int64_t T_lat = x->ne[1], B = x->ne[2];
    const int mem = hparams_.dit_n_memory;

    // timestep features (expo fourier) -> MLP
    ggml_tensor *t_in = ggml_new_tensor_1d(c, GGML_TYPE_F32, B);
    t_in = ggml_fill_inplace(c, t_in, t_value);
    ggml_tensor *te = ggml_view_4d(c, t_in, 1, B, 1, 1, t_in->nb[0], t_in->nb[0], t_in->nb[0], 0);
    te = ggml_repeat_4d(c, te, timestep_freqs_->ne[0], B, 1, 1);
    ggml_tensor *args = ggml_mul(c, te, timestep_freqs_);     // [128,B]
    ggml_tensor *tf = ggml_concat(c, ggml_cos(c, args), ggml_sin(c, args), 0); // [256,B]
    tf = lin(c, to_tstep0_w_, to_tstep0_b_, tf);
    tf = ggml_silu(c, tf);
    ggml_tensor *t_embed = lin(c, to_tstep2_w_, to_tstep2_b_, tf);   // [E,B]

    // global embed = to_global(global_raw) + t_embed
    ggml_tensor *gp = ggml_mul_mat(c, to_global0_, global_raw);
    gp = ggml_silu(c, gp);
    gp = ggml_mul_mat(c, to_global2_, gp);
    ggml_tensor *global_embed = ggml_add(c, gp, t_embed);           // [E,B]
    // per-block modulation source g6 = scale_shift_gate(per block) + gce(global)
    ggml_tensor *g = lin(c, gce0_w_, gce0_b_, global_embed);
    g = ggml_silu(c, g);
    ggml_tensor *g6 = lin(c, gce2_w_, gce2_b_, g);                  // [6E,B]

    // context = to_cond(cross_raw) [E,Tc,B]
    ggml_tensor *ctx = ggml_mul_mat(c, to_cond0_, cross_raw);
    ctx = ggml_silu(c, ctx);
    ctx = ggml_mul_mat(c, to_cond2_, ctx);

    // x preprocess: conv1d k1 residual, then project_in, prepend memory tokens
    ggml_tensor *xp = conv1x1_res(c, pre_conv_, x);                 // [256,T_lat,B]
    ggml_tensor *h = ggml_mul_mat(c, project_in_, xp);             // [E,T_lat,B]
    ggml_tensor *mem_t = ggml_reshape_3d(c, memory_tokens_, E, mem, 1);
    mem_t = ggml_repeat_4d(c, mem_t, E, mem, B, 1);
    h = ggml_concat(c, mem_t, h, 1);                               // [E, mem+T_lat, B]
    const int64_t T_full = h->ne[1];

    // positions [T_full*B], pos = idx % T_full (filled by caller)
    ggml_tensor *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T_full * B);
    ggml_set_input(pos); ggml_set_name(pos, "sa3.dit_pos");
    pos_out = pos;
    ggml_tensor *posd = ggml_dup(c, pos);

    // add scale_shift_gate per block inside BuildDiTBlock via g6 + L.ssg.
    for (int i = 0; i < hparams_.dit_depth; ++i)
    {
        ggml_tensor *g6_i = ggml_add(c, g6, dit_blocks_[i].scale_shift_gate);
        h = BuildDiTBlock(c, dit_blocks_[i], h, ctx, g6_i, posd);
    }

    // drop memory tokens, project_out, postprocess conv
    h = ggml_cont(c, ggml_view_3d(c, h, E, T_lat, B, h->nb[1], h->nb[2], (size_t)mem*h->nb[1]));
    ggml_tensor *o = ggml_mul_mat(c, project_out_, h);            // [256,T_lat,B]
    o = conv1x1_res(c, post_conv_, o);
    return o;
}

// =====================================================================
// RunDiffusion — ping-pong rectified flow (host-stepped).
// =====================================================================

bool StableAudio3Model::RunDiffusion(StableAudio3State &s, ggml_backend_sched_t sched)
{
    const int latent = hparams_.latent_dim;
    const int cond = hparams_.cond_token_dim;
    const double lps = (double)hparams_.audio_sample_rate / hparams_.same_downsample;
    int T_lat = (int)std::lround(s.duration_s * lps);
    if (T_lat < 2) T_lat = 2;
    if (T_lat & 1) ++T_lat;          // SAME-S needs even T_lat

    // CFG batch=2 when we have real text: slot0 = conditional (text), slot1 =
    // unconditional (prompt padding embedding). Otherwise single unconditional.
    const bool have_text = s.n_text_tokens > 0 && !s.text_cond.empty();
    const int  B = have_text ? 2 : 1;
    const int  T_txt = have_text ? s.n_text_tokens : 1;

    ggml_backend_t be = pick_backend(sched);
    const size_t CTX = (size_t)SA3_MAX_NODES * (sizeof(ggml_tensor) + 320);

    // persistent device inputs: x (shared across CFG slots), cross_raw, global_raw
    ggml_backend_buffer_t pbuf = nullptr; ggml_context *pctx = nullptr;
    ggml_tensor *x_dev = nullptr, *cross_dev = nullptr, *global_dev = nullptr;
    auto freep = [&](){ if (pbuf) ggml_backend_buffer_free(pbuf); if (pctx) ggml_free(pctx); pbuf=nullptr; pctx=nullptr; };
    {
        ggml_init_params ip = { 4*ggml_tensor_overhead() + (1<<16), nullptr, true };
        pctx = ggml_init(ip);
        x_dev = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, latent, T_lat, B);
        cross_dev = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, cond, T_txt, B);
        global_dev = ggml_new_tensor_2d(pctx, GGML_TYPE_F32, cond, B);
        ggml_set_name(x_dev, "sa3.x"); ggml_set_name(cross_dev, "sa3.cross"); ggml_set_name(global_dev, "sa3.global");
        pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
        if (!pbuf) { RS_LOG_ERR("[SA3] persistent alloc failed\n"); ggml_free(pctx); return false; }
    }

    // cross_raw: slot0 = T5 text_cond [cond, T_txt]; slot1 = padding embedding
    // broadcast over T_txt (unconditional). With no text, single padding token.
    {
        std::vector<float> pad(cond, 0.f);
        if (prompt_pad_) ggml_backend_tensor_get(prompt_pad_, pad.data(), 0, cond*sizeof(float));
        std::vector<float> hc((size_t)cond * T_txt * B, 0.f);
        if (have_text)
        {
            // slot0 = text_cond (already [cond, T_txt] row-major: token-major)
            std::memcpy(hc.data(), s.text_cond.data(),
                        std::min(s.text_cond.size(), (size_t)cond*T_txt)*sizeof(float));
            // slot1 = padding embedding for every position
            for (int tk = 0; tk < T_txt; ++tk)
                std::memcpy(hc.data() + (size_t)cond*T_txt + (size_t)tk*cond, pad.data(), cond*sizeof(float));
        }
        else
        {
            std::memcpy(hc.data(), pad.data(), cond*sizeof(float));
        }
        ggml_backend_tensor_set(cross_dev, hc.data(), 0, hc.size()*sizeof(float));
    }
    // global_raw = seconds_total embedding of duration (expo fourier + linear).
    {
        std::vector<float> hg((size_t)cond * B, 0.f);
        if (seconds_w_ && seconds_b_)
        {
            const int half = hparams_.timestep_feat / 2;     // 128
            std::vector<float> fr(half);
            ggml_backend_tensor_get(timestep_freqs_, fr.data(), 0, half*sizeof(float));
            float norm = (std::min(std::max(s.duration_s, hparams_.seconds_min), hparams_.seconds_max) - hparams_.seconds_min)
                         / (hparams_.seconds_max - hparams_.seconds_min);
            std::vector<float> ff(2*half);
            for (int i = 0; i < half; ++i) { float a = norm*fr[i]; ff[i]=std::cos(a); ff[half+i]=std::sin(a); }
            // seconds_w_ ggml ne=[256,768] (torch [768,256]); y = W @ ff + b
            std::vector<float> W((size_t)2*half*cond), bb(cond);
            ggml_backend_tensor_get(seconds_w_, W.data(), 0, W.size()*sizeof(float));
            ggml_backend_tensor_get(seconds_b_, bb.data(), 0, bb.size()*sizeof(float));
            std::vector<float> out(cond);
            for (int o = 0; o < cond; ++o) { float acc = bb[o]; for (int i = 0; i < 2*half; ++i) acc += W[(size_t)o*(2*half)+i]*ff[i]; out[o]=acc; }
            for (int b2 = 0; b2 < B; ++b2) std::memcpy(hg.data()+(size_t)b2*cond, out.data(), cond*sizeof(float));
        }
        ggml_backend_tensor_set(global_dev, hg.data(), 0, hg.size()*sizeof(float));
    }

    // host x init: N(0,1)
    std::vector<float> x((size_t)latent * T_lat);
    { std::normal_distribution<float> nd(0.f,1.f); for (auto &z : x) z = nd(s.rng()); }

    const int N = hparams_.n_sample_steps;
    const size_t slot = (size_t)latent * T_lat;
    std::vector<float> vbuf(slot * B);
    std::vector<float> xrep(slot * B);
    for (int i = 0; i < N; ++i)
    {
        const float t_curr = t_schedule_[i], t_next = t_schedule_[i+1];
        for (int bb = 0; bb < B; ++bb) std::memcpy(xrep.data()+(size_t)bb*slot, x.data(), slot*sizeof(float));
        ggml_backend_tensor_set(x_dev, xrep.data(), 0, xrep.size()*sizeof(float));

        ggml_init_params ip = { CTX, nullptr, true };
        ggml_context *c = ggml_init(ip);
        ggml_cgraph *gf = ggml_new_graph_custom(c, SA3_MAX_NODES, false);
        ggml_tensor *pos = nullptr;
        ggml_tensor *v = BuildDiTForward(c, x_dev, t_curr, cross_dev, global_dev, pos);
        ggml_build_forward_expand(gf, v);

        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gf)) { RS_LOG_ERR("[SA3] dit alloc (i=%d)\n", i); ggml_free(c); freep(); return false; }
        if (pos)
        {
            const int64_t n = pos->ne[0], Tf = n / B;
            std::vector<int32_t> p(n); for (int64_t j=0;j<n;++j) p[j] = (int32_t)(j % Tf);
            ggml_backend_tensor_set(pos, p.data(), 0, n*sizeof(int32_t));
        }
        if (sa3_sched_compute(sched, gf, &imatrix_cb_) != GGML_STATUS_SUCCESS) { RS_LOG_ERR("[SA3] dit compute (i=%d)\n", i); ggml_free(c); freep(); return false; }
        ggml_backend_tensor_get(v, vbuf.data(), 0, vbuf.size()*sizeof(float));
        ggml_free(c); ggml_backend_sched_reset(sched);

        // CFG combine: v = v_uncond + cfg*(v_cond - v_uncond). slot0=cond, slot1=uncond.
        std::vector<float> vc(slot);
        if (B == 2)
        {
            const float w = hparams_.cfg_scale;
            for (size_t j = 0; j < slot; ++j) vc[j] = vbuf[slot+j] + w * (vbuf[j] - vbuf[slot+j]);
        }
        else std::memcpy(vc.data(), vbuf.data(), slot*sizeof(float));

        std::vector<float> denoised(slot);
        for (size_t j = 0; j < slot; ++j) denoised[j] = x[j] - t_curr * vc[j];
        if (i < N-1 && t_next > 0.f)
        {
            std::normal_distribution<float> nd(0.f,1.f);
            for (size_t j = 0; j < x.size(); ++j) x[j] = (1.f-t_next)*denoised[j] + t_next*nd(s.rng());
        }
        else x = denoised;
    }
    freep();

    s.latents = x;
    s.n_latent_frames = T_lat;
    RS_LOG_INFO("[SA3] diffusion latents [%d, %d]\n", latent, T_lat);
    if (const char *p = std::getenv("RS_SA3_DUMP_LATENTS"))
        if (FILE *f = std::fopen(p, "wb")) { int32_t h[2]={T_lat,latent}; std::fwrite(h,4,2,f); std::fwrite(x.data(),4,x.size(),f); std::fclose(f); }
    return true;
}

// =====================================================================
// SAME-S decoder block (differential attention + DyT). x:[dim,T,M].
// =====================================================================

ggml_tensor *StableAudio3Model::BuildSameBlock(ggml_context *c, const SA3SameBlock &L,
                                               ggml_tensor *x) const
{
    const int dim = hparams_.same_dim, nh = hparams_.same_n_head;
    const int hd = hparams_.same_head_dim, rd = hparams_.same_rope_dims;
    const float scale = 1.0f / std::sqrt((float)hd);
    const int64_t T = x->ne[1], M = x->ne[2];

    // positions for this block (created as input, filled by caller via dup chain
    // is awkward; instead we bake arange%T using a fresh input). We reuse one pos
    // per block — caller fills all of them identically. To keep the graph simple
    // we create the pos here and register via name; RunSameDecoder fills every
    // tensor named "sa3.same_pos.*".
    ggml_tensor *pos = ggml_new_tensor_1d(c, GGML_TYPE_I32, T * M);
    ggml_set_input(pos); ggml_set_name(pos, "sa3.same_pos");

    ggml_tensor *h = dyt(c, x, L.pre_norm);
    ggml_tensor *qkv = ggml_mul_mat(c, L.to_qkv, h);          // [5*dim,T,M]
    auto part = [&](int i){ return ggml_cont(c, ggml_view_3d(c, qkv, dim, T, M, qkv->nb[1], qkv->nb[2], (size_t)i*dim*qkv->nb[0])); };
    ggml_tensor *q = to_heads(c, part(0), hd, nh);
    ggml_tensor *k = to_heads(c, part(1), hd, nh);
    ggml_tensor *v = to_heads(c, part(2), hd, nh);
    ggml_tensor *qd = to_heads(c, part(3), hd, nh);
    ggml_tensor *kd = to_heads(c, part(4), hd, nh);
    q  = dyt(c, q,  L.q_norm);  k  = dyt(c, k,  L.k_norm);
    qd = dyt(c, qd, L.q_norm);  kd = dyt(c, kd, L.k_norm);
    q  = rope_heads(c, q,  pos, rd);  k  = rope_heads(c, k,  pos, rd);
    qd = rope_heads(c, qd, pos, rd);  kd = rope_heads(c, kd, pos, rd);
    ggml_tensor *o1 = sdpa(c, q,  k,  v, scale);
    ggml_tensor *o2 = sdpa(c, qd, kd, v, scale);
    ggml_tensor *o = ggml_sub(c, o1, o2);
    o = ggml_mul_mat(c, L.to_out, o);
    x = ggml_add(c, x, o);

    ggml_tensor *hf = dyt(c, x, L.ff_norm);
    hf = swiglu(c, hf, L.ff_glu_w, L.ff_glu_b, L.ff_out_w, L.ff_out_b, hparams_.same_ff_inner);
    return ggml_add(c, x, hf);
}

bool StableAudio3Model::RunSameDecoder(StableAudio3State &s, ggml_backend_sched_t sched)
{
    if (!same_loaded_) { RS_LOG_ERR("[SA3] SAME decoder weights not loaded\n"); return false; }
    const int dim = hparams_.same_dim, latent = hparams_.latent_dim;
    const int stride = hparams_.same_stride;            // 16
    const int sub = stride + 1;                         // 17
    const int eff = 2 * sub;                            // 34 (chunk size)
    const int shift = eff / 2;                          // 17
    const int ch = hparams_.audio_channels, patch = hparams_.patch_size; // 2, 256
    const int outc = hparams_.same_out_ch;             // 512
    const int T_lat = s.n_latent_frames;
    if (T_lat < 2 || (T_lat & 1)) { RS_LOG_ERR("[SA3] SAME needs even T_lat (got %d)\n", T_lat); return false; }

    ggml_backend_t be = pick_backend(sched);
    const size_t CTX = (size_t)SA3_MAX_NODES * (sizeof(ggml_tensor) + 320);

    ggml_backend_buffer_t pbuf = nullptr; ggml_context *pctx = nullptr;
    ggml_tensor *lat_dev = nullptr;
    auto freep = [&](){ if (pbuf) ggml_backend_buffer_free(pbuf); if (pctx) ggml_free(pctx); };
    {
        ggml_init_params ip = { 2*ggml_tensor_overhead()+(1<<16), nullptr, true };
        pctx = ggml_init(ip);
        lat_dev = ggml_new_tensor_3d(pctx, GGML_TYPE_F32, latent, T_lat, 1);
        ggml_set_name(lat_dev, "sa3.same_lat");
        pbuf = ggml_backend_alloc_ctx_tensors(pctx, be);
        if (!pbuf) { RS_LOG_ERR("[SA3] same alloc failed\n"); ggml_free(pctx); return false; }
    }
    ggml_backend_tensor_set(lat_dev, s.latents.data(), 0, s.latents.size()*sizeof(float));

    ggml_init_params ip = { CTX, nullptr, true };
    ggml_context *c = ggml_init(ip);
    ggml_cgraph *gf = ggml_new_graph_custom(c, SA3_MAX_NODES, false);

    // running_std scalar bottleneck, project_in 256->768.
    ggml_tensor *x = ggml_mul(c, lat_dev, same_running_std_);      // broadcast scalar
    x = lin(c, same_proj_in_w_, same_proj_in_b_, x);              // [768,T_lat,1]

    // append 16 new_tokens per latent: build [768, 17, T_lat] then flatten.
    ggml_tensor *x_e = ggml_reshape_4d(c, x, dim, 1, T_lat, 1);
    ggml_tensor *nt = ggml_reshape_3d(c, same_new_tokens_, dim, 1, 1);
    nt = ggml_repeat_4d(c, nt, dim, stride, T_lat, 1);            // [768,16,T_lat,1]
    ggml_tensor *seq = ggml_concat(c, x_e, nt, 1);               // [768,17,T_lat,1]
    int internal_T = T_lat * sub;                                 // 17*T_lat
    x = ggml_reshape_3d(c, ggml_cont(c, seq), dim, internal_T, 1);

    // phase 1: chunks of 34, blocks 0..2
    int nc1 = internal_T / eff;
    x = ggml_reshape_3d(c, x, dim, eff, nc1);                    // [768,34,nc1]
    for (int i = 0; i < 3; ++i) x = BuildSameBlock(c, same_blocks_[i], x);
    x = ggml_reshape_3d(c, ggml_cont(c, x), dim, internal_T, 1);

    // shift by 17 both ends, blocks 3..5
    ggml_tensor *left  = ggml_cont(c, ggml_view_3d(c, x, dim, shift, 1, x->nb[1], x->nb[2], 0));
    ggml_tensor *right = ggml_cont(c, ggml_view_3d(c, x, dim, shift, 1, x->nb[1], x->nb[2], (size_t)(internal_T-shift)*x->nb[1]));
    x = ggml_concat(c, ggml_concat(c, left, x, 1), right, 1);    // [768, internal_T+34, 1]
    int padded_T = internal_T + eff;
    int nc2 = padded_T / eff;
    x = ggml_reshape_3d(c, x, dim, eff, nc2);
    for (int i = 3; i < 6; ++i) x = BuildSameBlock(c, same_blocks_[i], x);
    x = ggml_reshape_3d(c, ggml_cont(c, x), dim, padded_T, 1);
    x = ggml_cont(c, ggml_view_3d(c, x, dim, internal_T, 1, x->nb[1], x->nb[2], (size_t)shift*x->nb[1]));

    // drop the latent slot (keep 16 new-token positions per latent)
    x = ggml_reshape_3d(c, x, dim, sub, T_lat);                  // [768,17,T_lat]
    x = ggml_cont(c, ggml_view_3d(c, x, dim, stride, T_lat, x->nb[1], x->nb[2], (size_t)1*x->nb[1])); // drop idx0
    x = ggml_reshape_3d(c, x, dim, stride * T_lat, 1);           // [768, 16*T_lat, 1]

    // mapping conv 768->512 k3 p1
    ggml_tensor *patches = conv1d_k3p1(c, same_map_w_, same_map_b_, x); // [512, 16*T_lat, 1]

    ggml_build_forward_expand(gf, patches);

    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) { RS_LOG_ERR("[SA3] same alloc graph\n"); ggml_free(c); freep(); return false; }
    // fill all same_pos inputs (every block created one): pos[idx]=idx%34.
    for (ggml_tensor *t = ggml_get_first_tensor(c); t; t = ggml_get_next_tensor(c, t))
    {
        if (std::strcmp(ggml_get_name(t), "sa3.same_pos") == 0 && t->buffer)
        {
            const int64_t n = t->ne[0];
            std::vector<int32_t> p(n); for (int64_t j=0;j<n;++j) p[j]=(int32_t)(j % eff);
            ggml_backend_tensor_set(t, p.data(), 0, n*sizeof(int32_t));
        }
    }
    if (sa3_sched_compute(sched, gf, &imatrix_cb_) != GGML_STATUS_SUCCESS) { RS_LOG_ERR("[SA3] same compute\n"); ggml_free(c); freep(); return false; }

    const int L16 = stride * T_lat;
    std::vector<float> pbufh((size_t)outc * L16);
    ggml_backend_tensor_get(patches, pbufh.data(), 0, pbufh.size()*sizeof(float));
    ggml_free(c); freep(); ggml_backend_sched_reset(sched);

    if (const char *p = std::getenv("RS_SA3_DUMP_DEC"))
        if (FILE *f = std::fopen(p, "wb")) { int32_t h[2]={L16,outc}; std::fwrite(h,4,2,f); std::fwrite(pbufh.data(),4,pbufh.size(),f); std::fclose(f); }

    // patched_decode: 'b (c h) l -> b c (l h)', h=patch=256, c=2.
    // pbufh is ggml [outc=512, L16]: channel axis (ne0) is fastest, so element
    // (ch_idx, l) is at index ch_idx + outc*l. Channel index = c*patch + h.
    const int N = L16 * patch;                  // samples per channel
    s.audio_output.assign((size_t)N * ch, 0.f);  // interleaved stereo
    for (int l = 0; l < L16; ++l)
        for (int cc = 0; cc < ch; ++cc)
            for (int hh = 0; hh < patch; ++hh)
            {
                float val = pbufh[(size_t)(cc*patch + hh) + (size_t)outc * l];
                long n = (long)l*patch + hh;          // sample index
                s.audio_output[(size_t)n*ch + cc] = val;
            }
    RS_LOG_INFO("[SA3] decoded %d samples/ch (%d ch) @ %d Hz\n", N, ch, hparams_.audio_sample_rate);
    return true;
}

// =====================================================================
// Static registration
// =====================================================================

extern void rs_register_model_arch(
    const std::string &arch, std::function<std::shared_ptr<ISpeechModel>()> creator);

namespace {
struct StableAudio3Registrar
{
    StableAudio3Registrar()
    {
        auto creator = []() -> std::shared_ptr<ISpeechModel> { return std::make_shared<StableAudio3Model>(); };
        rs_register_model_arch("stable-audio3", creator);
        rs_register_model_arch("stable_audio_3", creator);
    }
}
#if defined(__GNUC__) || defined(__clang__)
__attribute__((used))
#endif
global_stable_audio3_reg;
}  // namespace
