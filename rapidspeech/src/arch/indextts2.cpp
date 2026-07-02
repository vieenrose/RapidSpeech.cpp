// IndexTTS-2 GGUF tensor binding + arch registration.
//
// Forward graphs (GPT-2 AR, S2Mel DiT+WaveNet, BigVGAN-v2, w2v-bert encoder)
// are bound here. This file:
//   - reads hparams from GGUF metadata,
//   - walks every `indextts2.*` tensor and stores it in the right pointer,
//   - runs the IndexTTS2 encode/decode pipeline end-to-end.
//
// Tensor name → struct field mapping mirrors what
// scripts/convert_indextts2_to_gguf.py writes.

#include "indextts2.h"

#include "arch/campplus.h"
#include "arch/semantic_codec.h"
#include "arch/w2v_bert.h"
#include "core/rs_context.h"
#include "frontend/seamless_mel.h"
#include "gguf.h"
#include "utils/rs_log.h"
#include "utils/rs_wav.h"
#ifdef RS_USE_METAL_BIGVGAN
#include "indextts2_bigvgan_metal.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <regex>
#include <sstream>
#include <string>
#include <algorithm>

namespace indextts2 {

namespace {

static std::string prod_dump_dir() {
    const char *d = std::getenv("RS_INDEXTTS2_DUMP_PROD_DIR");
    if (!d || !*d) return {};
    std::string s = d;
    if (!s.empty() && s.back() != '/') s.push_back('/');
    return s;
}

static void dump_f32_if_enabled(const std::string &dir, const char *name,
                                const std::vector<float> &v) {
    if (dir.empty() || v.empty()) return;
    FILE *f = std::fopen((dir + name + ".f32").c_str(), "wb");
    if (!f) return;
    std::fwrite(v.data(), sizeof(float), v.size(), f);
    std::fclose(f);
}

} // namespace

// ---------------------------------------------------------------------------
// Tokenizer: implementation lives in indextts2_tokenizer.cpp. The vocab +
// scores are loaded from GGUF metadata at Load() time below.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Small GGUF helpers
// ---------------------------------------------------------------------------
static int32_t gguf_u32_or(gguf_context *ctx, const char *key, int32_t dflt) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return dflt;
    return (int32_t)gguf_get_val_u32(ctx, i);
}

static std::string gguf_str_or(gguf_context *ctx, const char *key,
                               const std::string &dflt) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return dflt;
    return gguf_get_val_str(ctx, i);
}

// ---------------------------------------------------------------------------
// Ctor / dtor
// ---------------------------------------------------------------------------
Model::Model() {
    meta_.arch_name         = "indextts2";
    meta_.audio_sample_rate = 22050;     // BigVGAN-v2 22kHz target
    meta_.n_mels            = 80;
    meta_.vocab_size        = 12000;
    meta_.use_external_frontend = true;  // we do our own mel + w2v-bert
}

Model::~Model() {
    if (bigvgan_.buf)      ggml_backend_buffer_free(bigvgan_.buf);
    if (bigvgan_.ctx_data) ggml_free(bigvgan_.ctx_data);
    if (bigvgan_.ctx_gguf) gguf_free(bigvgan_.ctx_gguf);
}

// ---------------------------------------------------------------------------
// Load: bind every `indextts2.*` tensor by name
// ---------------------------------------------------------------------------
bool Model::LoadHParamsFromGGUF(gguf_context *g) {
    hp_.n_layer       = gguf_u32_or(g, "indextts2.gpt.n_layer",       hp_.n_layer);
    hp_.n_embd        = gguf_u32_or(g, "indextts2.gpt.n_embd",        hp_.n_embd);
    hp_.n_head        = gguf_u32_or(g, "indextts2.gpt.n_head",        hp_.n_head);
    hp_.head_dim      = hp_.n_embd / hp_.n_head;
    hp_.ff_dim        = 4 * hp_.n_embd;
    hp_.text_vocab    = gguf_u32_or(g, "indextts2.gpt.text_vocab",    hp_.text_vocab);
    hp_.mel_codes     = gguf_u32_or(g, "indextts2.gpt.mel_codes",     hp_.mel_codes);
    hp_.start_mel     = gguf_u32_or(g, "indextts2.gpt.start_mel",     hp_.start_mel);
    hp_.stop_mel      = gguf_u32_or(g, "indextts2.gpt.stop_mel",      hp_.stop_mel);
    hp_.max_text_tok  = gguf_u32_or(g, "indextts2.gpt.max_text_tokens", hp_.max_text_tok);
    hp_.max_mel_tok   = gguf_u32_or(g, "indextts2.gpt.max_mel_tokens",  hp_.max_mel_tok);
    hp_.cond_blocks   = gguf_u32_or(g, "indextts2.gpt.cond_blocks",   hp_.cond_blocks);
    hp_.cond_heads    = gguf_u32_or(g, "indextts2.gpt.cond_heads",    hp_.cond_heads);
    hp_.cond_dim      = gguf_u32_or(g, "indextts2.gpt.cond_dim",      hp_.cond_dim);
    hp_.cond_ff       = gguf_u32_or(g, "indextts2.gpt.cond_ff",       hp_.cond_ff);
    hp_.cond_latents  = gguf_u32_or(g, "indextts2.gpt.cond_latents",  hp_.cond_latents);
    hp_.emo_blocks    = gguf_u32_or(g, "indextts2.gpt.emo_blocks",    hp_.emo_blocks);
    hp_.emo_heads     = gguf_u32_or(g, "indextts2.gpt.emo_heads",     hp_.emo_heads);
    hp_.emo_ff        = gguf_u32_or(g, "indextts2.gpt.emo_ff",        hp_.emo_ff);
    hp_.s2mel_sr            = gguf_u32_or(g, "indextts2.s2mel.sr",          hp_.s2mel_sr);
    hp_.s2mel_n_fft         = gguf_u32_or(g, "indextts2.s2mel.n_fft",       hp_.s2mel_n_fft);
    hp_.s2mel_hop           = gguf_u32_or(g, "indextts2.s2mel.hop",         hp_.s2mel_hop);
    hp_.s2mel_win           = gguf_u32_or(g, "indextts2.s2mel.win",         hp_.s2mel_win);
    hp_.s2mel_n_mels        = gguf_u32_or(g, "indextts2.s2mel.n_mels",      hp_.s2mel_n_mels);
    hp_.s2mel_dit_depth     = gguf_u32_or(g, "indextts2.s2mel.dit_depth",   hp_.s2mel_dit_depth);
    hp_.s2mel_dit_hidden    = gguf_u32_or(g, "indextts2.s2mel.dit_hidden",  hp_.s2mel_dit_hidden);
    hp_.s2mel_dit_heads     = gguf_u32_or(g, "indextts2.s2mel.dit_heads",   hp_.s2mel_dit_heads);
    hp_.s2mel_wavenet_layers = gguf_u32_or(g, "indextts2.s2mel.wavenet_layers", hp_.s2mel_wavenet_layers);
    hp_.s2mel_wavenet_kernel = gguf_u32_or(g, "indextts2.s2mel.wavenet_kernel", hp_.s2mel_wavenet_kernel);
    hp_.s2mel_lr_channels    = gguf_u32_or(g, "indextts2.s2mel.lr_channels",    hp_.s2mel_lr_channels);
    hp_.s2mel_lr_in_channels = gguf_u32_or(g, "indextts2.s2mel.lr_in_channels", hp_.s2mel_lr_in_channels);
    hp_.s2mel_style_dim      = gguf_u32_or(g, "indextts2.s2mel.style_dim",      hp_.s2mel_style_dim);
    hp_.sc_vocab        = gguf_u32_or(g, "indextts2.semantic_codec.vocab",        hp_.sc_vocab);
    hp_.sc_hidden       = gguf_u32_or(g, "indextts2.semantic_codec.hidden",       hp_.sc_hidden);
    hp_.sc_codebook_dim = gguf_u32_or(g, "indextts2.semantic_codec.codebook_dim", hp_.sc_codebook_dim);

    meta_.vocab_size = hp_.text_vocab;
    meta_.audio_sample_rate = hp_.bigvgan_sr;
    meta_.n_mels = hp_.s2mel_n_mels;
    blocks_.assign(hp_.n_layer, GPT2Block{});
    cond_enc_.blocks.assign(hp_.cond_blocks, ConformerBlock{});
    cond_enc_.n_blocks = hp_.cond_blocks;
    emo_cond_enc_.blocks.assign(hp_.emo_blocks, ConformerBlock{});
    emo_cond_enc_.n_blocks = hp_.emo_blocks;
    return true;
}

// Tries to parse a GPT-2 block index from "h.<L>.<rest>" and writes the
// tensor into the matching field. Returns true if it matched.
static bool try_bind_gpt_block(std::vector<GPT2Block> &blocks,
                               const std::string &key, ggml_tensor *t) {
    if (key.rfind("h.", 0) != 0) return false;
    size_t dot = key.find('.', 2);
    if (dot == std::string::npos) return false;
    int L = std::atoi(key.c_str() + 2);
    if (L < 0 || L >= (int)blocks.size()) return false;
    const std::string sub = key.substr(dot + 1);
    auto &b = blocks[L];
    if (sub == "ln_1.weight")        { b.ln1_w = t; return true; }
    if (sub == "ln_1.bias")          { b.ln1_b = t; return true; }
    if (sub == "attn.c_attn.weight") { b.attn_qkv_w = t; return true; }
    if (sub == "attn.c_attn.bias")   { b.attn_qkv_b = t; return true; }
    if (sub == "attn.c_proj.weight") { b.attn_proj_w = t; return true; }
    if (sub == "attn.c_proj.bias")   { b.attn_proj_b = t; return true; }
    if (sub == "ln_2.weight")        { b.ln2_w = t; return true; }
    if (sub == "ln_2.bias")          { b.ln2_b = t; return true; }
    if (sub == "mlp.c_fc.weight")    { b.mlp_fc_w = t; return true; }
    if (sub == "mlp.c_fc.bias")      { b.mlp_fc_b = t; return true; }
    if (sub == "mlp.c_proj.weight")  { b.mlp_proj_w = t; return true; }
    if (sub == "mlp.c_proj.bias")    { b.mlp_proj_b = t; return true; }
    return false;
}

// Parses "<root>.<layer_idx>.<rest>" — returns layer_idx or -1.
static int parse_layer_index(const std::string &key, std::string &rest) {
    size_t dot = key.find('.');
    if (dot == std::string::npos) return -1;
    int L = std::atoi(key.c_str());
    if (L == 0 && key[0] != '0') return -1;
    rest = key.substr(dot + 1);
    return L;
}

bool Model::LoadGPT(struct ggml_context *gguf_data) {
    const std::string prefix = "indextts2.gpt.";
    int n_bound = 0;
    for (ggml_tensor *t = ggml_get_first_tensor(gguf_data); t != nullptr;
         t = ggml_get_next_tensor(gguf_data, t)) {
        const std::string name = ggml_get_name(t);
        if (name.rfind(prefix, 0) != 0) continue;
        const std::string key = name.substr(prefix.size());

        // Embeddings + final projections (HF GPT-2 + IndexTTS-specific heads)
        if (key == "text_embedding.weight")     { text_emb_w_     = t; n_bound++; continue; }
        if (key == "mel_embedding.weight")      { mel_emb_w_      = t; n_bound++; continue; }
        if (key == "text_pos_embedding.emb.weight") { text_pos_emb_w_ = t; n_bound++; continue; }
        if (key == "mel_pos_embedding.emb.weight")  { mel_pos_emb_w_  = t; n_bound++; continue; }
        if (key == "gpt.wpe.weight")           { gpt_wpe_w_    = t; n_bound++; continue; }
        if (key == "gpt.ln_f.weight")          { gpt_ln_f_w_  = t; n_bound++; continue; }
        if (key == "gpt.ln_f.bias")            { gpt_ln_f_b_  = t; n_bound++; continue; }
        if (key == "final_norm.weight")        { final_norm_w_ = t; n_bound++; continue; }
        if (key == "final_norm.bias")          { final_norm_b_ = t; n_bound++; continue; }
        if (key == "mel_head.weight")          { mel_head_w_  = t; n_bound++; continue; }
        if (key == "mel_head.bias")            { mel_head_b_  = t; n_bound++; continue; }
        if (key == "text_head.weight")         { text_head_w_ = t; n_bound++; continue; }
        if (key == "text_head.bias")           { text_head_b_ = t; n_bound++; continue; }
        if (key == "emovec_layer.weight") { emovec_layer_w_ = t; n_bound++; continue; }
        if (key == "emovec_layer.bias")   { emovec_layer_b_ = t; n_bound++; continue; }
        if (key == "emo_layer.weight")    { emo_layer_w_    = t; n_bound++; continue; }
        if (key == "emo_layer.bias")      { emo_layer_b_    = t; n_bound++; continue; }
        if (key == "speed_emb.weight")    { speed_emb_w_    = t; n_bound++; continue; }

        // GPT-2 transformer body lives under "gpt.h.<L>...".
        if (key.rfind("gpt.h.", 0) == 0) {
            if (try_bind_gpt_block(blocks_, key.substr(strlen("gpt.")), t)) {
                n_bound++; continue;
            }
        }

        // Conformer encoders — keyed by raw subkey (forward looks them up).
        if (key.rfind("conditioning_encoder.", 0) == 0) {
            const std::string sub = key.substr(strlen("conditioning_encoder."));
            // Body is "encoders.<L>.<rest>" in espnet-style Conformer.
            if (sub.rfind("encoders.", 0) == 0) {
                std::string rest;
                int L = parse_layer_index(sub.substr(strlen("encoders.")), rest);
                if (L >= 0 && L < (int)cond_enc_.blocks.size()) {
                    cond_enc_.blocks[L].by_name[rest] = t;
                    n_bound++; continue;
                }
            }
            cond_enc_.by_name[sub] = t;
            n_bound++; continue;
        }
        if (key.rfind("emo_conditioning_encoder.", 0) == 0) {
            const std::string sub = key.substr(strlen("emo_conditioning_encoder."));
            if (sub.rfind("encoders.", 0) == 0) {
                std::string rest;
                int L = parse_layer_index(sub.substr(strlen("encoders.")), rest);
                if (L >= 0 && L < (int)emo_cond_enc_.blocks.size()) {
                    emo_cond_enc_.blocks[L].by_name[rest] = t;
                    n_bound++; continue;
                }
            }
            emo_cond_enc_.by_name[sub] = t;
            n_bound++; continue;
        }

        // Perceiver resamplers (depth=2; layers.<L>.0=Attention, .1=FF).
        auto bind_perceiver = [&](PerceiverResampler &p,
                                  const std::string &sub) -> bool {
            if (sub == "latents")               { p.latents = t; return true; }
            if (sub == "proj_context.weight")   { p.proj_ctx_w = t; return true; }
            if (sub == "proj_context.bias")     { p.proj_ctx_b = t; return true; }
            if (sub == "norm.gamma" || sub == "norm.weight") {
                p.norm_gamma = t; return true;
            }
            if (sub.rfind("layers.", 0) == 0) {
                std::string rest;
                int L = parse_layer_index(sub.substr(strlen("layers.")), rest);
                if (L < 0) return false;
                if (L >= (int)p.layers.size()) p.layers.resize(L + 1);
                p.layers[L].by_name[rest] = t;
                return true;
            }
            return false;
        };
        if (key.rfind("perceiver_encoder.", 0) == 0) {
            if (bind_perceiver(cond_perceiver_,
                               key.substr(strlen("perceiver_encoder.")))) {
                n_bound++; continue;
            }
        }
        if (key.rfind("emo_perceiver_encoder.", 0) == 0) {
            if (bind_perceiver(emo_cond_perceiver_,
                               key.substr(strlen("emo_perceiver_encoder.")))) {
                n_bound++; continue;
            }
        }

        // Unknown — log once at debug level so forward authors can spot gaps.
        RS_LOG_DEBUG("[indextts2] unbound gpt tensor: %s", key.c_str());
    }
    RS_LOG_INFO("[indextts2] gpt bound %d tensors", n_bound);
    if (n_bound == 0) {
        RS_LOG_WARN("[indextts2] no GPT tensors in GGUF — AR forward will be "
                    "a no-op (this is expected during bring-up before "
                    "gpt.pth is converted).");
    }
    return true;
}

bool Model::LoadS2Mel(struct ggml_context *gguf_data) {
    const std::string prefix = "indextts2.s2mel.";
    int n = 0;
    for (ggml_tensor *t = ggml_get_first_tensor(gguf_data); t != nullptr;
         t = ggml_get_next_tensor(gguf_data, t)) {
        const std::string name = ggml_get_name(t);
        if (name.rfind(prefix, 0) != 0) continue;
        const std::string key = name.substr(prefix.size());

        if (key == "gpt_layer.0.weight") { s2mel_.gpt_layer0_w = t; n++; continue; }
        if (key == "gpt_layer.0.bias")   { s2mel_.gpt_layer0_b = t; n++; continue; }
        if (key == "gpt_layer.1.weight") { s2mel_.gpt_layer1_w = t; n++; continue; }
        if (key == "gpt_layer.1.bias")   { s2mel_.gpt_layer1_b = t; n++; continue; }
        if (key == "gpt_layer.2.weight") { s2mel_.gpt_layer2_w = t; n++; continue; }
        if (key == "gpt_layer.2.bias")   { s2mel_.gpt_layer2_b = t; n++; continue; }

        if (key.rfind("length_regulator.", 0) == 0) {
            s2mel_.length_regulator[key.substr(strlen("length_regulator."))] = t;
            n++; continue;
        }
        if (key.rfind("cfm.", 0) == 0) {
            s2mel_.cfm[key.substr(strlen("cfm."))] = t;
            n++; continue;
        }
        RS_LOG_DEBUG("[indextts2] unbound s2mel tensor: %s", key.c_str());
    }
    RS_LOG_INFO("[indextts2] s2mel bound %d tensors", n);
    return true;
}

bool Model::LoadSemanticCodec(struct ggml_context *gguf_data, gguf_context *ctx_gguf,
                               ggml_backend_t backend) {
    semantic_codec_model_ = std::make_shared<semantic_codec::SemanticCodecModel>();
    if (!semantic_codec_model_->Load(gguf_data, ctx_gguf, backend,
                                      "indextts2.semantic_codec.")) {
        RS_LOG_ERR("[indextts2] SemanticCodecModel::Load failed");
        semantic_codec_model_.reset();
        return false;
    }
    RS_LOG_INFO("[indextts2] semantic_codec loaded (hidden=%d, codebook=%dx%d)",
                semantic_codec_model_->hparams().hidden,
                semantic_codec_model_->hparams().codebook_size,
                semantic_codec_model_->hparams().codebook_dim);
    return true;
}

bool Model::LoadCampplus(struct ggml_context *gguf_data, gguf_context *ctx_gguf,
                          ggml_backend_t backend) {
    campplus_model_ = std::make_shared<CAMPPlusModel>();
    if (!campplus_model_->LoadDirect(gguf_data, ctx_gguf, backend,
                                      "indextts2.campplus.")) {
        RS_LOG_ERR("[indextts2] CAMPPlusModel::LoadDirect failed");
        campplus_model_.reset();
        return false;
    }
    RS_LOG_INFO("[indextts2] campplus loaded via CAMPPlusModel (prefix=indextts2.campplus.)");
    return true;
}

bool Model::LoadW2VBert(struct ggml_context *gguf_data, gguf_context *ctx_gguf,
                          ggml_backend_t backend) {
    w2v_bert_model_ = std::make_shared<w2v_bert::W2VBertModel>();
    if (!w2v_bert_model_->Load(gguf_data, ctx_gguf, backend,
                                "indextts2.w2v_bert.")) {
        RS_LOG_ERR("[indextts2] W2VBertModel::Load failed");
        w2v_bert_model_.reset();
        return false;
    }
    RS_LOG_INFO("[indextts2] w2v_bert loaded (layers=%d, hidden=%d)",
                w2v_bert_model_->hparams().n_layers,
                w2v_bert_model_->hparams().hidden);
    return true;
}

bool Model::LoadStaticMatrices(struct ggml_context *gguf_data) {
    for (ggml_tensor *t = ggml_get_first_tensor(gguf_data); t != nullptr;
         t = ggml_get_next_tensor(gguf_data, t)) {
        const std::string name = ggml_get_name(t);
        if (name == "indextts2.spk_matrix") spk_matrix_ = t;
        if (name == "indextts2.emo_matrix") emo_matrix_ = t;
        if (name == "indextts2.w2v.mean")   w2v_mean_   = t;
        if (name == "indextts2.w2v.std")    w2v_std_    = t;
    }
    return true;
}

bool Model::Load(const std::unique_ptr<rs_context_t> &ctx,
                 ggml_backend_t backend) {
    backend_     = backend;
    cpu_backend_ = backend; // unified for now; BigVGAN may rebind later
    if (!LoadHParamsFromGGUF(ctx->ctx_gguf)) return false;

    if (!LoadGPT(ctx->gguf_data))            return false;
    if (!LoadS2Mel(ctx->gguf_data))          return false;
    if (!LoadSemanticCodec(ctx->gguf_data, ctx->ctx_gguf, backend)) return false;
    if (!LoadCampplus(ctx->gguf_data, ctx->ctx_gguf, backend)) return false;
    if (!LoadW2VBert(ctx->gguf_data, ctx->ctx_gguf, backend)) return false;
    LoadStaticMatrices(ctx->gguf_data);

    // Tokenizer vocab + scores live in GGUF metadata (convert script emits
    // them via sentencepiece-py). Falls back to byte-level encoding if absent.
    {
        int64_t k_tok = gguf_find_key(ctx->ctx_gguf, "tokenizer.ggml.tokens");
        int64_t k_sco = gguf_find_key(ctx->ctx_gguf, "tokenizer.ggml.scores");
        if (k_tok >= 0) {
            const int64_t n = gguf_get_arr_n(ctx->ctx_gguf, k_tok);
            std::vector<std::string> tokens;
            tokens.reserve((size_t)n);
            for (int64_t i = 0; i < n; ++i)
                tokens.emplace_back(gguf_get_arr_str(ctx->ctx_gguf, k_tok, i));
            std::vector<float> scores;
            if (k_sco >= 0) {
                const float *sd = (const float *)gguf_get_arr_data(
                    ctx->ctx_gguf, k_sco);
                const int64_t ns = gguf_get_arr_n(ctx->ctx_gguf, k_sco);
                scores.assign(sd, sd + ns);
            }
            bpe_.load_from_vocab(std::move(tokens), std::move(scores));
            RS_LOG_INFO("[indextts2] tokenizer: %zu pieces loaded from GGUF",
                        bpe_.id_to_token.size());
        } else {
            RS_LOG_WARN("[indextts2] tokenizer.ggml.tokens not in GGUF — "
                        "falling back to byte-level encoding");
        }
    }

    RS_LOG_INFO("[indextts2] loaded: %d layers, %d embd, %d mel codes, "
                "%d sr, %d mels",
                hp_.n_layer, hp_.n_embd, hp_.mel_codes, hp_.bigvgan_sr,
                hp_.s2mel_n_mels);
    RS_LOG_INFO("[indextts2] forward graphs ready (AR / S2Mel / BigVGAN)");
    return true;
}

std::shared_ptr<RSState> Model::CreateState() {
    auto s = std::make_shared<State>();
    s->rng.seed((uint32_t)seed_);
    return s;
}

// ---------------------------------------------------------------------------
// TTS interface — minimal stubs that route inputs into State
// ---------------------------------------------------------------------------
bool Model::PushText(RSState &state, const char *text,
                     const char *language, const char *instruct) {
    auto &s = static_cast<State &>(state);
    if (text)    s.text_in   = text;
    if (language) s.language = language;
    if (instruct) s.instruct = instruct;
    // Real BPE encoding lands with the forward CL.
    s.text_token_ids = bpe_.encode(s.text_in);
    {
        std::ostringstream oss;
        const size_t n_show = std::min<size_t>(s.text_token_ids.size(), 32);
        for (size_t i = 0; i < n_show; ++i) {
            if (i) oss << ",";
            oss << s.text_token_ids[i];
        }
        if (s.text_token_ids.size() > n_show) oss << ",...";
        RS_LOG_INFO("[indextts2] text tokens: n=%zu ids=[%s]",
                    s.text_token_ids.size(), oss.str().c_str());
    }
    return true;
}

bool Model::PushReferenceAudio(RSState &state, const float *samples,
                               int n_samples, int sample_rate,
                               ggml_backend_sched_t sched) {
    auto &s = static_cast<State &>(state);
    const std::string dump_dir = prod_dump_dir();
    const int max_ref_samples = sample_rate > 0 ? sample_rate * 15 : n_samples;
    const int n_ref = std::max(0, std::min(n_samples, max_ref_samples));
    s.ref_audio.assign(samples, samples + n_ref);
    s.ref_sr = sample_rate;

    // Resample to 16 kHz for CAMPPlus + W2V-BERT (both expect 16k mono PCM).
    const float *pcm16 = s.ref_audio.data();
    int n16 = (int)s.ref_audio.size();
    std::vector<float> resampled;
    if (sample_rate != 16000) {
        if (!resample_pcm(s.ref_audio, sample_rate, resampled, 16000)) {
            RS_LOG_ERR("[indextts2] PushReferenceAudio: resample %d→16000 failed",
                       sample_rate);
            return false;
        }
        pcm16 = resampled.data();
        n16   = (int)resampled.size();
    }

    // Resample to 22.05 kHz for the S2Mel prompt mel.
    if (sample_rate == hp_.s2mel_sr) {
        s.ref_audio_22k = s.ref_audio;
    } else {
        s.ref_audio_22k.clear();
        if (!resample_pcm(s.ref_audio, sample_rate, s.ref_audio_22k,
                          hp_.s2mel_sr)) {
            RS_LOG_ERR("[indextts2] PushReferenceAudio: resample %d→%d failed",
                       sample_rate, hp_.s2mel_sr);
            return false;
        }
    }

    // CAMPPlus → 192-d speaker embedding into s.spk_embedding.
    if (campplus_model_) {
        CAMPPlusState cs;
        if (!campplus_model_->Embed(pcm16, n16, cs, sched,
                                    /*l2_normalize=*/false)) {
            RS_LOG_ERR("[indextts2] CAMPPlus::Embed failed");
            return false;
        }
        s.spk_embedding = std::move(cs.embedding);
        RS_LOG_INFO("[indextts2] CAMPPlus spk_embedding ready (dim=%zu)",
                    s.spk_embedding.size());
    } else {
        RS_LOG_WARN("[indextts2] PushReferenceAudio: campplus_model_ not loaded");
    }

    // W2V-BERT + semantic_codec → sc_hidden for AR conditioning.
    if (!ComputeScHidden(pcm16, n16, s.sc_hidden, s.T_sc, sched,
                         /*is_emo=*/false)) {
        RS_LOG_WARN("[indextts2] PushReferenceAudio: ComputeScHidden failed — "
                    "set RS_INDEXTTS2_SC_HIDDEN_NPY for pre-computed features");
        s.sc_hidden.clear();
        s.T_sc = 0;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ComputeScHidden: fbank → W2V-BERT → per-channel normalize → semantic_codec
// quantize. `pcm16` is 16 kHz mono. Shared by the speaker and emotion audio
// paths. Mirrors the spk_cond_emb / S_ref computation in infer_v2.py.
// ---------------------------------------------------------------------------
bool Model::ComputeScHidden(const float *pcm16, int n16,
                            std::vector<float> &out, int &T_out,
                            ggml_backend_sched_t sched, bool is_emo) {
    out.clear();
    T_out = 0;
    if (!(w2v_bert_model_ && semantic_codec_model_ && w2v_mean_ && w2v_std_)) {
        return false;
    }
    const std::string dump_dir = prod_dump_dir();

    // Step 1: fbank-160 extraction.
    SeamlessMelExtractor mel_ext;
    std::vector<float> fbank;
    std::vector<float> pcm_vec(pcm16, pcm16 + n16);
    int T_fbank = mel_ext.Compute(pcm_vec, fbank);
    if (T_fbank <= 0) {
        RS_LOG_ERR("[indextts2] SeamlessMelExtractor failed (n16=%d)", n16);
        return false;
    }
    RS_LOG_INFO("[indextts2] fbank: %d samples → T=%d", n16, T_fbank);
    dump_f32_if_enabled(dump_dir, is_emo ? "emo_w2v_input_features"
                                         : "w2v_input_features", fbank);

    // Step 2: W2V-BERT forward.
    const int D = w2v_bert_model_->hparams().hidden; // 1024
    std::vector<float> hidden17((size_t)T_fbank * D);
    if (!w2v_bert_model_->Forward(fbank.data(), T_fbank,
                                   nullptr, hidden17.data(), sched)) {
        RS_LOG_ERR("[indextts2] W2V-BERT forward failed");
        return false;
    }

    // Step 3: Per-channel normalize (feat - mean) / std.
    std::vector<float> w2v_mean(D), w2v_std(D);
    ggml_backend_tensor_get(w2v_mean_, w2v_mean.data(), 0, D * sizeof(float));
    ggml_backend_tensor_get(w2v_std_,  w2v_std.data(),  0, D * sizeof(float));
    for (int t = 0; t < T_fbank; ++t) {
        for (int d = 0; d < D; ++d) {
            float &v = hidden17[(size_t)t * D + d];
            v = (v - w2v_mean[d]) / w2v_std[d];
        }
    }
    dump_f32_if_enabled(dump_dir, is_emo ? "emo_cond_emb" : "spk_cond_emb",
                        hidden17);

    // Step 4: Semantic codec quantize.
    out.resize((size_t)T_fbank * D);
    if (!semantic_codec_model_->Forward(hidden17.data(), T_fbank,
                                         out.data(), nullptr, sched)) {
        RS_LOG_ERR("[indextts2] semantic_codec forward failed");
        out.clear();
        return false;
    }
    T_out = T_fbank;
    dump_f32_if_enabled(dump_dir, is_emo ? "emo_S_ref" : "S_ref", out);
    RS_LOG_INFO("[indextts2] %s sc_hidden ready (T=%d D=%d)",
                is_emo ? "emo" : "spk", T_fbank, D);
    return true;
}

bool Model::PushReferenceText(RSState &state, const char *ref_text) {
    (void)state; (void)ref_text;
    // IndexTTS-2 does not use a reference transcript directly — emotion is
    // optionally driven by a separate prompt routed through QwenEmotion.
    return true;
}

// ---------------------------------------------------------------------------
// PushEmotionAudio: independent emotion reference audio (emo_control_method=1).
// Only feeds the emo-conditioning path (get_emovec) → State::emo_sc_hidden; the
// speaker timbre / S2Mel prompt / CAMPPlus style stay from PushReferenceAudio.
// ---------------------------------------------------------------------------
bool Model::PushEmotionAudio(RSState &state, const float *samples,
                             int n_samples, int sample_rate,
                             ggml_backend_sched_t sched) {
    auto &s = static_cast<State &>(state);
    if (!samples || n_samples <= 0) {
        s.emo_sc_hidden.clear();
        s.emo_T_sc = 0;
        return false;
    }
    // Upstream cuts the emotion reference to 15 s (sr=16000 path).
    const int max_ref_samples = sample_rate > 0 ? sample_rate * 15 : n_samples;
    const int n_ref = std::max(0, std::min(n_samples, max_ref_samples));
    std::vector<float> emo_audio(samples, samples + n_ref);

    // Resample to 16 kHz for W2V-BERT + semantic_codec.
    const float *pcm16 = emo_audio.data();
    int n16 = (int)emo_audio.size();
    std::vector<float> resampled;
    if (sample_rate != 16000) {
        if (!resample_pcm(emo_audio, sample_rate, resampled, 16000)) {
            RS_LOG_ERR("[indextts2] PushEmotionAudio: resample %d→16000 failed",
                       sample_rate);
            return false;
        }
        pcm16 = resampled.data();
        n16   = (int)resampled.size();
    }

    if (!ComputeScHidden(pcm16, n16, s.emo_sc_hidden, s.emo_T_sc, sched,
                         /*is_emo=*/true)) {
        RS_LOG_WARN("[indextts2] PushEmotionAudio: ComputeScHidden failed");
        s.emo_sc_hidden.clear();
        s.emo_T_sc = 0;
        return false;
    }
    RS_LOG_INFO("[indextts2] emo_sc_hidden ready (T=%d)", s.emo_T_sc);
    return true;
}

// normalize_emo_vec: mirrors infer_v2.py IndexTTS2.normalize_emo_vec.
// Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm].
std::vector<float> Model::normalize_emo_vec(const std::vector<float> &vec8,
                                            bool apply_bias) {
    std::vector<float> v = vec8;
    v.resize(8, 0.0f);
    if (apply_bias) {
        static const float bias[8] = {0.9375f, 0.875f, 1.0f, 1.0f,
                                      0.9375f, 0.9375f, 0.6875f, 0.5625f};
        for (int i = 0; i < 8; ++i) v[i] *= bias[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < 8; ++i) sum += v[i];
    if (sum > 0.8f) {
        const float scale = 0.8f / sum;
        for (int i = 0; i < 8; ++i) v[i] *= scale;
    }
    return v;
}

void Model::SetEmotionControl(RSState &state, int mode, float emo_alpha,
                              const float *vec8, bool use_random,
                              bool apply_bias, const char *emo_text) {
    auto &s = static_cast<State &>(state);
    s.emo_mode       = mode;
    s.emo_alpha      = emo_alpha;
    s.emo_use_random = use_random;
    s.emo_apply_bias = apply_bias;
    s.emo_text       = emo_text ? emo_text : "";
    if (vec8) {
        std::vector<float> v(vec8, vec8 + 8);
        // Mode 2 (custom vectors) applies the bias de-emphasis up front, exactly
        // as webui.py gen_single does before calling infer().
        s.emo_vector = (mode == 2) ? normalize_emo_vec(v, apply_bias) : v;
    } else {
        s.emo_vector.clear();
    }
    RS_LOG_INFO("[indextts2] emotion control: mode=%d alpha=%.3f random=%d "
                "has_vec=%d emo_text='%s'", mode, emo_alpha, (int)use_random,
                (int)!s.emo_vector.empty(), s.emo_text.c_str());
}

// ---------------------------------------------------------------------------
// ResolveEmotion: assemble the final emo conditioning vector (`emovec`,
// [n_embd]) per State::emo_mode. Mirrors infer_v2.py infer_generator emotion
// handling (steps 0–6 + per-segment merge). See the plan/Context for the
// authoritative ordering.
// ---------------------------------------------------------------------------
bool Model::ResolveEmotion(State &s, ggml_backend_sched_t sched,
                           std::vector<float> &emovec) {
    const int D = hp_.n_embd; // 1280
    emovec.assign((size_t)D, 0.0f);

    if (s.sc_hidden.empty() || s.T_sc <= 0) {
        // No speaker conditioning at all → zero emo (voice-agnostic AR).
        return true;
    }
    const int D_in = hp_.sc_hidden;
    if ((int)s.sc_hidden.size() != s.T_sc * D_in) {
        RS_LOG_WARN("[indextts2] ResolveEmotion: spk sc_hidden size mismatch");
        return true;
    }

    // Step (B): mode 3 → QwenEmotion(text) → 8-d vector.
    if (s.emo_mode == 3 && s.emo_vector.empty()) {
        const std::string et = s.emo_text.empty() ? s.text_in : s.emo_text;
        std::vector<float> qv;
        if (QwenEmoInfer(sched, et, qv) && qv.size() == 8) {
            s.emo_vector = qv;
            RS_LOG_INFO("[indextts2] QwenEmotion vec: [%.3f %.3f %.3f %.3f "
                        "%.3f %.3f %.3f %.3f]", qv[0], qv[1], qv[2], qv[3],
                        qv[4], qv[5], qv[6], qv[7]);
        } else {
            RS_LOG_WARN("[indextts2] QwenEmoInfer unavailable — emo text ignored");
        }
    }

    const bool using_vector = (s.emo_vector.size() == 8);

    // Step (C): scale the 8-d vector by clamp(emo_alpha,0,1), truncated to 4dp.
    std::vector<float> wvec;
    if (using_vector) {
        const float a = std::max(0.0f, std::min(1.0f, s.emo_alpha));
        wvec = s.emo_vector;
        for (float &x : wvec)
            x = (float)((long)(x * a * 10000.0f)) / 10000.0f;
    }

    // Steps (A)+(D): pick the emo source sc_hidden and the merge alpha.
    //   vector/text mode → emo_audio=None → use speaker audio, merge alpha=1.0
    //   external emo audio (mode 1, no vector) → emo_sc_hidden, merge alpha=user
    //   otherwise (mode 0) → speaker audio, merge alpha=1.0
    const std::vector<float> *emo_src = &s.sc_hidden;
    int emo_T = s.T_sc;
    float merge_alpha = 1.0f;
    if (!using_vector && !s.emo_sc_hidden.empty() && s.emo_T_sc > 0 &&
        (int)s.emo_sc_hidden.size() == s.emo_T_sc * D_in) {
        emo_src = &s.emo_sc_hidden;
        emo_T   = s.emo_T_sc;
        merge_alpha = s.emo_alpha;
    }

    // Per-segment merge: base from speaker, emo from the chosen source.
    std::vector<float> base_vec, emo_vec;
    if (!RunGetEmovec(s, sched, s.sc_hidden, s.T_sc, base_vec)) return false;
    if (emo_src == &s.sc_hidden) {
        emo_vec = base_vec; // identical input → identical get_emovec
    } else if (!RunGetEmovec(s, sched, *emo_src, emo_T, emo_vec)) {
        return false;
    }
    for (int j = 0; j < D; ++j)
        emovec[j] = base_vec[j] + merge_alpha * (emo_vec[j] - base_vec[j]);

    // Vector path: emovec = emovec_mat + (1 - Σw) * emovec.
    std::vector<float> emovec_mat_dump;
    if (using_vector) {
        std::vector<float> emovec_mat;
        if (ComputeEmovecMat(s, wvec, emovec_mat) && (int)emovec_mat.size() == D) {
            emovec_mat_dump = emovec_mat;
            float wsum = 0.0f;
            for (float w : wvec) wsum += w;
            for (int j = 0; j < D; ++j)
                emovec[j] = emovec_mat[j] + (1.0f - wsum) * emovec[j];
        } else {
            RS_LOG_WARN("[indextts2] ResolveEmotion: emovec_mat unavailable — "
                        "using audio-derived emovec only");
        }
    }

    if (const char *dir_c = std::getenv("RS_INDEXTTS2_EMO_TEST_DIR")) {
        std::string dir = dir_c;
        if (!dir.empty() && dir.back() != '/') dir.push_back('/');
        dump_f32_if_enabled(dir, "base_vec", base_vec);
        dump_f32_if_enabled(dir, "emo_vec_src", emo_vec);
        dump_f32_if_enabled(dir, "emovec", emovec);
        dump_f32_if_enabled(dir, "style", s.spk_embedding);
        dump_f32_if_enabled(dir, "wvec", wvec);
        dump_f32_if_enabled(dir, "emovec_mat", emovec_mat_dump);
    }
    return true;
}

// ComputeEmovecMat: the emo_matrix vector path (infer_v2.py lines 480-491).
//   for each of 8 emotion groups: pick a prototype index (cosine of the speaker
//   CAMPPlus style against spk_matrix rows, or random), gather emo_matrix row,
//   then weighted-sum by `wvec` → [n_embd].
bool Model::ComputeEmovecMat(State &s, const std::vector<float> &wvec,
                             std::vector<float> &out) {
    out.clear();
    if (!spk_matrix_ || !emo_matrix_) return false;
    const int spk_dim = (int)spk_matrix_->ne[0];   // 192 (style dim)
    const int n_proto = (int)spk_matrix_->ne[1];
    const int emo_dim = (int)emo_matrix_->ne[0];   // 1280
    if ((int)emo_matrix_->ne[1] != n_proto) {
        RS_LOG_WARN("[indextts2] emovec_mat: spk/emo matrix row mismatch (%d vs %lld)",
                    n_proto, (long long)emo_matrix_->ne[1]);
        return false;
    }
    if (emo_dim != hp_.n_embd) {
        RS_LOG_WARN("[indextts2] emovec_mat: emo_matrix dim %d != n_embd %d",
                    emo_dim, hp_.n_embd);
        return false;
    }
    const std::vector<int> &groups = hp_.emo_num;
    int gsum = 0;
    for (int g : groups) gsum += g;
    if (gsum != n_proto || (int)groups.size() != 8) {
        RS_LOG_WARN("[indextts2] emovec_mat: Σemo_num=%d != n_proto=%d", gsum,
                    n_proto);
        return false;
    }

    // Host copies of the matrices (f32).
    std::vector<float> spk_mat((size_t)spk_dim * n_proto);
    std::vector<float> emo_mat((size_t)emo_dim * n_proto);
    ggml_backend_tensor_get(spk_matrix_, spk_mat.data(), 0,
                            spk_mat.size() * sizeof(float));
    ggml_backend_tensor_get(emo_matrix_, emo_mat.data(), 0,
                            emo_mat.size() * sizeof(float));

    // CAMPPlus style of the speaker reference (192-d). Falls back to row 0 of
    // each group when unavailable / wrong dim (still produces a usable vector).
    const bool have_style = ((int)s.spk_embedding.size() == spk_dim);
    const float *style = have_style ? s.spk_embedding.data() : nullptr;
    double style_norm = 0.0;
    if (style) {
        for (int d = 0; d < spk_dim; ++d) style_norm += (double)style[d] * style[d];
        style_norm = std::sqrt(style_norm) + 1e-12;
    }

    out.assign((size_t)emo_dim, 0.0f);
    int start = 0;
    std::uniform_int_distribution<int> *unused = nullptr; (void)unused;
    for (int g = 0; g < 8; ++g) {
        const int gn = groups[g];
        int idx = 0; // local index within the group
        if (s.emo_use_random) {
            std::uniform_int_distribution<int> dist(0, gn - 1);
            idx = dist(s.rng);
        } else if (style) {
            float best = -2.0f;
            for (int r = 0; r < gn; ++r) {
                const float *row = &spk_mat[(size_t)(start + r) * spk_dim];
                double dot = 0.0, rn = 0.0;
                for (int d = 0; d < spk_dim; ++d) {
                    dot += (double)style[d] * row[d];
                    rn  += (double)row[d] * row[d];
                }
                float cos = (float)(dot / (style_norm * (std::sqrt(rn) + 1e-12)));
                if (cos > best) { best = cos; idx = r; }
            }
        }
        const float *sel = &emo_mat[(size_t)(start + idx) * emo_dim];
        const float w = (g < (int)wvec.size()) ? wvec[g] : 0.0f;
        for (int d = 0; d < emo_dim; ++d) out[d] += w * sel[d];
        start += gn;
    }
    return true;
}

int Model::GetAudioOutput(RSState &state, float **out_data) {
    auto &s = static_cast<State &>(state);
    if (s.audio_output.empty()) { *out_data = nullptr; return 0; }
    *out_data = s.audio_output.data();
    return (int)s.audio_output.size();
}

// ---------------------------------------------------------------------------
// W2V-BERT 2.0 smoke harness
// ---------------------------------------------------------------------------
bool Model::RunW2VBertSmoke(State &s, ggml_backend_sched_t sched, bool &taken) {
    (void)s;
    taken = false;
    const char *dir_c = std::getenv("RS_INDEXTTS2_W2V_BERT_TEST_DIR");
    if (!dir_c) return true;
    std::string dir = dir_c;
    if (!dir.empty() && dir.back() != '/') dir.push_back('/');
    taken = true;

    if (!w2v_bert_model_) {
        RS_LOG_ERR("[w2v-smoke] W2V-BERT model not loaded");
        return false;
    }

    // ---- Fbank smoke: load audio_16k.npy → SeamlessMelExtractor → dump input_features_cpp.f32 ----
    {
        std::string path = dir + "audio_16k.npy";
        FILE *f = std::fopen(path.c_str(), "rb");
        if (f) {
            // Parse .npy header.
            char magic[10];
            if (std::fread(magic, 1, 10, f) == 10 &&
                std::memcmp(magic, "\x93NUMPY", 6) == 0) {
                uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
                std::vector<char> hdr(hlen);
                std::fread(hdr.data(), 1, hlen, f);
                std::string hs(hdr.begin(), hdr.end());
                // Find number of samples.
                size_t shp = hs.find("'shape':");
                size_t lp  = hs.find('(', shp);
                size_t rp  = hs.find(')', lp);
                std::string ss = hs.substr(lp + 1, rp - lp - 1);
                while (!ss.empty() && ss.front() == ' ') ss.erase(ss.begin());
                while (!ss.empty() && ss.back()  == ' ') ss.pop_back();
                // 1D → just one integer (or comma-separated for 1-element tuple)
                size_t comma = ss.find(',');
                if (comma == std::string::npos) {
                    int N_samp = std::atoi(ss.c_str());
                    std::vector<float> pcm(N_samp);
                    if (std::fread(pcm.data(), sizeof(float), N_samp, f) == (size_t)N_samp) {
                        SeamlessMelExtractor mel_ext;
                        std::vector<float> fbank_out;
                        int T_out = mel_ext.Compute(pcm, fbank_out);
                        RS_LOG_INFO("[w2v-smoke] fbank: %d samples → T=%d (160-dim)", N_samp, T_out);

                        // Dump input_features_cpp.f32.
                        FILE *fd = std::fopen((dir + "input_features_cpp.f32").c_str(), "wb");
                        if (fd) {
                            std::fwrite(fbank_out.data(), sizeof(float), fbank_out.size(), fd);
                            std::fclose(fd);
                            RS_LOG_INFO("[w2v-smoke] dumped input_features_cpp.f32 [%d, 160]", T_out);
                        }
                    }
                }
            }
            std::fclose(f);
        }
    }

    // ---- Load input_features.npy [T, 160] ----
    std::vector<int> sh;
    std::vector<float> input_feat;
    {
        std::string path = dir + "input_features.npy";
        FILE *f = std::fopen(path.c_str(), "rb");
        if (!f) { RS_LOG_ERR("[w2v-smoke] cannot open %s", path.c_str()); return false; }
        char magic[10];
        if (std::fread(magic, 1, 10, f) != 10 ||
            std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            RS_LOG_ERR("[w2v-smoke] %s: not .npy", path.c_str());
            std::fclose(f); return false;
        }
        uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
        std::vector<char> hdr(hlen);
        std::fread(hdr.data(), 1, hlen, f);
        std::string hs(hdr.begin(), hdr.end());
        if (hs.find("'<f4'") == std::string::npos ||
            hs.find("'fortran_order': False") == std::string::npos) {
            RS_LOG_ERR("[w2v-smoke] %s: must be C-order f32", path.c_str());
            std::fclose(f); return false;
        }
        size_t shp = hs.find("'shape':");
        size_t lp  = hs.find('(', shp);
        size_t rp  = hs.find(')', lp);
        std::string ss = hs.substr(lp + 1, rp - lp - 1);
        size_t p = 0;
        while (p < ss.size()) {
            size_t c = ss.find(',', p);
            std::string tok = ss.substr(p, c == std::string::npos
                                            ? std::string::npos : c - p);
            while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
            while (!tok.empty() && tok.back()  == ' ') tok.pop_back();
            if (!tok.empty()) sh.push_back(std::atoi(tok.c_str()));
            if (c == std::string::npos) break;
            p = c + 1;
        }
        if (sh.size() != 2) {
            RS_LOG_ERR("[w2v-smoke] expected 2D input, got %zuD", sh.size());
            std::fclose(f); return false;
        }
        int T_in = sh[0], D_feat = sh[1];
        size_t n = (size_t)T_in * D_feat;
        input_feat.assign(n, 0);
        if (std::fread(input_feat.data(), sizeof(float), n, f) != n) {
            RS_LOG_ERR("[w2v-smoke] short read on %s", path.c_str());
            std::fclose(f); return false;
        }
        std::fclose(f);
        RS_LOG_INFO("[w2v-smoke] loaded input_features [%d, %d]", T_in, D_feat);
    }

    int T = sh[0];
    int D = w2v_bert_model_->hparams().hidden;

    // ---- Run forward (outputs both hidden0 and hidden17) ----
    std::vector<float> h0((size_t)T * D);
    std::vector<float> h17((size_t)T * D);
    if (!w2v_bert_model_->Forward(input_feat.data(), T,
                                   h0.data(), h17.data(), sched)) {
        RS_LOG_ERR("[w2v-smoke] W2V-BERT forward failed");
        return false;
    }

    // ---- Dump ----
    auto dump_f32 = [](const std::string &path, const std::vector<float> &v) {
        FILE *fd = std::fopen(path.c_str(), "wb");
        if (!fd) return false;
        std::fwrite(v.data(), sizeof(float), v.size(), fd);
        std::fclose(fd);
        return true;
    };
    if (!dump_f32(dir + "w2v_hidden0.f32",  h0))  { RS_LOG_ERR("[w2v-smoke] write hidden0 failed");  return false; }
    if (!dump_f32(dir + "w2v_hidden17.f32", h17)) { RS_LOG_ERR("[w2v-smoke] write hidden17 failed"); return false; }

    RS_LOG_INFO("[w2v-smoke] dumped hidden0 + hidden17 (T=%d D=%d)", T, D);
    RS_LOG_INFO("[w2v-smoke] run: python scripts/diff_w2v_bert.py %s", dir_c);
    return true;
}

bool Model::Encode(const std::vector<float> &input_frames, RSState &state,
                   ggml_backend_sched_t sched) {
    (void)input_frames; (void)state; (void)sched;
    return true;
}

bool Model::Decode(RSState &state, ggml_backend_sched_t sched) {
    auto &s = static_cast<State &>(state);

    // GPT prefill smoke path (env var): bypass everything and feed a saved
    // inputs_embeds.npy through the GPT-2 prefill graph, dumping
    // gpt_last_hidden.f32 / final_norm_out.f32 / mel_head_logits.f32 for
    // scripts/diff_gpt_prefill.py.
    {
        bool taken = false;
        if (!RunGPTPrefillSmoke(s, sched, taken)) return false;
        if (taken) return true;
    }

    // Conformer smoke path (env var RS_INDEXTTS2_CONFORMER_TEST_DIR):
    // load sc_hidden.npy → RunConditioning → dump conformer_out.f32 /
    // cond_latents.f32 for scripts/diff_conformer.py.
    if (const char *dir_c = std::getenv("RS_INDEXTTS2_CONFORMER_TEST_DIR")) {
        std::string dir = dir_c;
        if (!dir.empty() && dir.back() != '/') dir.push_back('/');
        // Load sc_hidden.npy [T_sc, 1024] — C-order, so row = time step.
        // We need to transpose to [D_in * T_sc] = [1024, T_sc] ggml layout.
        const int D_in = hp_.sc_hidden; // 1024
        // Re-use the load_npy helper from the GPT smoke section via a local copy.
        FILE *f = std::fopen((dir + "sc_hidden.npy").c_str(), "rb");
        if (!f) {
            RS_LOG_ERR("[conformer-smoke] cannot open sc_hidden.npy");
            return false;
        }
        char magic[10];
        if (std::fread(magic, 1, 10, f) != 10 ||
            std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            RS_LOG_ERR("[conformer-smoke] sc_hidden.npy not a .npy");
            std::fclose(f); return false;
        }
        uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
        std::vector<char> hdr(hlen);
        std::fread(hdr.data(), 1, hlen, f);
        std::string hs(hdr.begin(), hdr.end());
        // Parse shape [T_sc, D_in].
        size_t shp = hs.find("'shape':");
        size_t lp  = hs.find('(', shp);
        int T_sc = 0, dim1 = 0;
        std::sscanf(hs.c_str() + lp + 1, "%d, %d", &T_sc, &dim1);
        if (dim1 != D_in) {
            RS_LOG_ERR("[conformer-smoke] sc_hidden shape [%d,%d] bad D",
                       T_sc, dim1);
            std::fclose(f); return false;
        }
        std::vector<float> sc_hidden_TD((size_t)T_sc * D_in);
        std::fread(sc_hidden_TD.data(), sizeof(float), sc_hidden_TD.size(), f);
        std::fclose(f);

        // PT saves [T_sc, D_in] C-fast → D_in is fastest in memory.
        // RunConditioning expects ne[0]=D_in fastest, which matches PT memory
        // layout exactly — no transpose needed.
        std::vector<float> cond_latents;
        if (!RunConditioning(s, sched, sc_hidden_TD, T_sc, cond_latents)) {
            RS_LOG_ERR("[conformer-smoke] RunConditioning failed");
            return false;
        }
        // Also exercise the emo path on the same hidden (mirrors the PT dumper,
        // which uses sc_hidden as the emo input too). Dumps emo_conformer_out.f32 /
        // emo_latents_raw.f32 / emo_vec.f32 to the same dir.
        std::vector<float> emo_vec;
        if (!RunGetEmovec(s, sched, sc_hidden_TD, T_sc, emo_vec)) {
            RS_LOG_ERR("[conformer-smoke] RunGetEmovec failed");
            return false;
        }
        // RunConditioning already dumped conformer_out.f32 + cond_latents.f32
        // to dir when RS_INDEXTTS2_CONFORMER_TEST_DIR is set.
        RS_LOG_INFO("[conformer-smoke] done T_sc=%d T_enc=%d "
                    "cond_latents=[32,%d]",
                    T_sc, (T_sc - 3) / 2 + 1, hp_.n_embd);
        return true;
    }

    // W2V-BERT 2.0 smoke path (env var RS_INDEXTTS2_W2V_BERT_TEST_DIR):
    // load input_features.npy → RunW2VBertSmoke → dump w2v_hidden0.f32 /
    // w2v_hidden17.f32 for scripts/diff_w2v_bert.py.
    {
        bool taken = false;
        if (!RunW2VBertSmoke(s, sched, taken)) return false;
        if (taken) return true;
    }

    // S2Mel smoke path (env var): bypass AR, load lm_latent.npy +
    // prompt_condition.npy + ref_mel.npy + style.npy + target_lengths.npy
    // from RS_INDEXTTS2_S2MEL_TEST_DIR, run gpt_layer + length_regulator,
    // dump gpt_layer_out.f32 / lr_out.f32 alongside the references.
    // If a BigVGAN GGUF is also available (env or already loaded) AND the
    // smoke produced a real mel (s.mel_pred), fall through to BigVGAN so the
    // session yields synthesized audio end-to-end from baked conditioning.
    {
        bool taken = false;
        if (!RunS2MelSmoke(s, sched, taken)) return false;
        if (taken) {
            if (s.mel_pred.empty()) return true;
            if (!bigvgan_ready_) {
                if (const char *p = std::getenv("RS_INDEXTTS2_BIGVGAN_GGUF")) {
                    LoadBigVGAN(p, backend_);
                }
            }
            if (!bigvgan_ready_) {
                RS_LOG_WARN("[indextts2] s2mel-smoke: BigVGAN not loaded "
                            "(set RS_INDEXTTS2_BIGVGAN_GGUF) — mel dumped, "
                            "no audio");
                return true;
            }
            const int n_mels  = hp_.s2mel_n_mels;
            const int T_mel22 = (int)(s.mel_pred.size() / n_mels);
            std::vector<float> mel_nmT((size_t)n_mels * T_mel22);
            for (int t = 0; t < T_mel22; ++t) {
                for (int m = 0; m < n_mels; ++m) {
                    mel_nmT[(size_t)m * T_mel22 + t] =
                        s.mel_pred[(size_t)t * n_mels + m];
                }
            }
            return RunBigVGAN(s, sched, mel_nmT.data(), n_mels, T_mel22);
        }
    }

    // BigVGAN smoke path (env var): bypass AR + S2Mel and feed a numpy mel
    // dump straight to the vocoder. Used for bit-exact bring-up against
    // scripts/dump_bigvgan_ref.py output. Lazily loads the bigvgan GGUF.
    if (const char *mel_path = std::getenv("RS_INDEXTTS2_BIGVGAN_TEST_MEL")) {
        if (!bigvgan_ready_) {
            if (const char *p = std::getenv("RS_INDEXTTS2_BIGVGAN_GGUF")) {
                LoadBigVGAN(p, backend_);
            }
        }
        if (!bigvgan_ready_) {
            RS_LOG_ERR("[indextts2] Decode: smoke path needs "
                       "RS_INDEXTTS2_BIGVGAN_GGUF to be set");
            return false;
        }
        FILE *f = std::fopen(mel_path, "rb");
        if (!f) {
            RS_LOG_ERR("[indextts2] Decode: cannot open mel test file %s",
                       mel_path);
            return false;
        }
        char magic[10];
        if (std::fread(magic, 1, 10, f) != 10 ||
            std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            RS_LOG_ERR("[indextts2] Decode: %s is not a .npy file", mel_path);
            std::fclose(f);
            return false;
        }
        uint16_t hlen = (uint8_t)magic[8] | ((uint8_t)magic[9] << 8);
        std::vector<char> hdr(hlen);
        if (std::fread(hdr.data(), 1, hlen, f) != hlen) {
            std::fclose(f);
            return false;
        }
        std::string hs(hdr.begin(), hdr.end());
        if (hs.find("'<f4'") == std::string::npos ||
            hs.find("'fortran_order': False") == std::string::npos) {
            RS_LOG_ERR("[indextts2] Decode: %s must be C-order float32",
                       mel_path);
            std::fclose(f);
            return false;
        }
        size_t shp = hs.find("'shape':");
        size_t lp  = hs.find('(', shp);
        int dim0 = 0, dim1 = 0;
        std::sscanf(hs.c_str() + lp + 1, "%d, %d", &dim0, &dim1);
        if (dim0 != hp_.s2mel_n_mels) {
            RS_LOG_ERR("[indextts2] Decode: mel.npy first dim %d != n_mels %d",
                       dim0, hp_.s2mel_n_mels);
            std::fclose(f);
            return false;
        }
        const int n_mels = dim0;
        const int T_mel  = dim1;
        std::vector<float> mel((size_t)n_mels * T_mel);
        if (std::fread(mel.data(), sizeof(float), mel.size(), f) != mel.size()) {
            RS_LOG_ERR("[indextts2] Decode: short read on %s", mel_path);
            std::fclose(f);
            return false;
        }
        std::fclose(f);
        RS_LOG_INFO("[indextts2] Decode: smoke path — loaded mel [%d, %d] from %s",
                    n_mels, T_mel, mel_path);
        if (!RunBigVGAN(s, sched, mel.data(), n_mels, T_mel)) return false;

        // Optional: dump wav.npy for off-line bit-exact diff against PyTorch.
        if (const char *out = std::getenv("RS_INDEXTTS2_BIGVGAN_DUMP_WAV")) {
            FILE *wf = std::fopen(out, "wb");
            if (wf) {
                const int64_t n = (int64_t)s.audio_output.size();
                std::fwrite(s.audio_output.data(), sizeof(float),
                            (size_t)n, wf);
                std::fclose(wf);
                RS_LOG_INFO("[indextts2] Decode: dumped %lld f32 samples → %s",
                            (long long)n, out);
            } else {
                RS_LOG_WARN("[indextts2] Decode: cannot open %s for dump", out);
            }
        }
        return true;
    }

    // 1. AR: produce mel codes (and, eventually, lm_latent).
    //
    // Optional bring-up: when RS_INDEXTTS2_SC_HIDDEN_NPY points to a
    // [T_sc, sc_hidden] C-fast f32 .npy, load it into State so RunAR runs the
    // real Conformer+Perceiver conditioning path instead of zero spk_latents /
    // emo_vec. Lets us drive end-to-end AR without waiting on W2V-BERT
    // (task #10).
    if (s.sc_hidden.empty()) {
        if (const char *p = std::getenv("RS_INDEXTTS2_SC_HIDDEN_NPY")) {
            FILE *f = std::fopen(p, "rb");
            if (!f) {
                RS_LOG_WARN("[indextts2] sc_hidden override: open failed: %s", p);
            } else {
                char magic[10];
                bool ok = (std::fread(magic, 1, 10, f) == 10 &&
                           std::memcmp(magic, "\x93NUMPY", 6) == 0);
                uint16_t hlen = ok ? (uint8_t)magic[8] | ((uint8_t)magic[9] << 8) : 0;
                std::vector<char> hdr(hlen);
                if (ok) ok = (std::fread(hdr.data(), 1, hlen, f) == hlen);
                if (ok) {
                    std::string hs(hdr.begin(), hdr.end());
                    if (hs.find("'<f4'") == std::string::npos ||
                        hs.find("'fortran_order': False") == std::string::npos) {
                        RS_LOG_WARN("[indextts2] sc_hidden override: must be "
                                    "C-order float32");
                        ok = false;
                    } else {
                        size_t shp = hs.find("'shape':");
                        size_t lp  = hs.find('(', shp);
                        int T_sc = 0, dim1 = 0;
                        std::sscanf(hs.c_str() + lp + 1, "%d, %d", &T_sc, &dim1);
                        if (dim1 != hp_.sc_hidden) {
                            RS_LOG_WARN("[indextts2] sc_hidden override: shape "
                                        "[%d,%d] bad D (expected %d)",
                                        T_sc, dim1, hp_.sc_hidden);
                            ok = false;
                        } else {
                            s.sc_hidden.resize((size_t)T_sc * dim1);
                            ok = (std::fread(s.sc_hidden.data(), sizeof(float),
                                             s.sc_hidden.size(), f) ==
                                  s.sc_hidden.size());
                            if (ok) {
                                s.T_sc = T_sc;
                                RS_LOG_INFO("[indextts2] sc_hidden override: "
                                            "loaded [%d, %d] from %s",
                                            T_sc, dim1, p);
                            } else {
                                s.sc_hidden.clear();
                                RS_LOG_WARN("[indextts2] sc_hidden override: "
                                            "short read on %s", p);
                            }
                        }
                    }
                }
                std::fclose(f);
            }
        }
    }
    if (!RunAR(s, sched)) {
        RS_LOG_ERR("[indextts2] Decode: RunAR failed");
        return false;
    }

    // 2. S2Mel: lm_latent + semantic codes → mel-spectrogram through
    //    the real CFM Euler solver and DiT/WaveNet estimator.
    if (!RunS2Mel(s, sched)) {
        RS_LOG_ERR("[indextts2] Decode: RunS2Mel failed");
        return false;
    }

    // 3. BigVGAN-v2 vocoder: separate GGUF, lazily loaded.
    if (!bigvgan_ready_) {
        if (const char *p = std::getenv("RS_INDEXTTS2_BIGVGAN_GGUF")) {
            LoadBigVGAN(p, backend_);
        }
    }
    if (!bigvgan_ready_) {
        RS_LOG_WARN("[indextts2] Decode: BigVGAN-v2 not yet loaded — audio "
                    "output is empty. Set RS_INDEXTTS2_BIGVGAN_GGUF.");
        return true;
    }

    // Production path: S2Mel-predicted mel → BigVGAN. RunS2Mel writes mel_pred
    // as (T_mel, n_mels) contiguous; RunBigVGAN expects (n_mels, T_mel) numpy
    // row-major (== ggml ne[0]=T_mel fastest), so we transpose here.
    if (s.mel_pred.empty()) {
        RS_LOG_WARN("[indextts2] Decode: mel_pred empty — skipping vocoder");
        return true;
    }
    const int n_mels  = hp_.s2mel_n_mels;
    const int T_mel22 = (int)(s.mel_pred.size() / n_mels);
    std::vector<float> mel_nmT((size_t)n_mels * T_mel22);
    for (int t = 0; t < T_mel22; ++t) {
        for (int m = 0; m < n_mels; ++m) {
            mel_nmT[(size_t)m * T_mel22 + t] = s.mel_pred[(size_t)t * n_mels + m];
        }
    }
    return RunBigVGAN(s, sched, mel_nmT.data(), n_mels, T_mel22);
}

bool Model::LoadBigVGAN(const std::string &path, ggml_backend_t backend) {
    if (path.empty() || !backend) return false;
    if (bigvgan_ready_) return true;

    struct gguf_init_params gp = { /*.no_alloc=*/true,
                                   /*.ctx=*/&bigvgan_.ctx_data };
    bigvgan_.ctx_gguf = gguf_init_from_file(path.c_str(), gp);
    if (!bigvgan_.ctx_gguf) {
        RS_LOG_ERR("[indextts2] BigVGAN: gguf_init_from_file failed: %s",
                   path.c_str());
        return false;
    }
    bigvgan_.buf = ggml_backend_alloc_ctx_tensors(bigvgan_.ctx_data, backend);
    if (!bigvgan_.buf) {
        RS_LOG_ERR("[indextts2] BigVGAN: tensor buffer alloc failed: %s",
                   path.c_str());
        return false;
    }

    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        RS_LOG_ERR("[indextts2] BigVGAN: fopen %s failed", path.c_str());
        return false;
    }
    const size_t data_off = gguf_get_data_offset(bigvgan_.ctx_gguf);
    const int64_t n_tensors = gguf_get_n_tensors(bigvgan_.ctx_gguf);
    std::vector<char> rb;
    int n_loaded = 0;
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *name = gguf_get_tensor_name(bigvgan_.ctx_gguf, i);
        ggml_tensor *t = ggml_get_tensor(bigvgan_.ctx_data, name);
        if (!t) continue;
        size_t t_off = gguf_get_tensor_offset(bigvgan_.ctx_gguf, i);
        size_t t_sz  = ggml_nbytes(t);
        if (t_sz == 0) continue;
        if (rb.size() < t_sz) rb.resize(t_sz);
        std::fseek(f, data_off + t_off, SEEK_SET);
        if (std::fread(rb.data(), 1, t_sz, f) != t_sz) {
            RS_LOG_ERR("[indextts2] BigVGAN: read failed for %s", name);
            std::fclose(f);
            return false;
        }
        ggml_backend_tensor_set(t, rb.data(), 0, t_sz);
        // Strip the "bigvgan." prefix so the forward path can index by
        // upstream-state-dict name (`conv_pre.weight`, `ups.0.0.weight`, …).
        const std::string nn = name;
        const std::string prefix = "indextts2.bigvgan.";
        const std::string key = (nn.rfind(prefix, 0) == 0)
                                    ? nn.substr(prefix.size())
                                    : nn;
        bigvgan_.tensors[key] = t;
        ++n_loaded;
    }
    std::fclose(f);
    bigvgan_ready_ = true;
    RS_LOG_INFO("[indextts2] BigVGAN: loaded %d tensors from %s",
                n_loaded, path.c_str());
#ifdef RS_USE_METAL_BIGVGAN
    // Fused Metal vocoder: bypasses the ggml graph (~467 CPU/Metal splits) with
    // a single command buffer. Falls back to the ggml path if init fails.
    {
        auto dec = std::make_unique<BigVGANMetalDecoder>();
        if (dec->init(bigvgan_, hp_)) {
            bigvgan_metal_ = std::move(dec);
            RS_LOG_INFO("[indextts2] BigVGAN: using fused Metal decoder");
        } else {
            RS_LOG_WARN("[indextts2] BigVGAN: Metal decoder init failed — "
                        "falling back to ggml graph");
        }
    }
#endif
    return true;
}

// Model::LoadQwenEmotion and Model::QwenEmoInfer are defined in
// indextts2_qwen_emo.cpp.

} // namespace indextts2

// ---------------------------------------------------------------------------
// Arch registration
// ---------------------------------------------------------------------------
extern void rs_register_model_arch(const std::string &arch,
                                   std::function<std::shared_ptr<ISpeechModel>()> creator);

namespace {
struct AutoRegister {
    AutoRegister() {
        auto creator = []() -> std::shared_ptr<ISpeechModel> {
            return std::make_shared<indextts2::Model>();
        };
        rs_register_model_arch("indextts2", creator);
        rs_register_model_arch("indextts-2", creator);
    }
};
static AutoRegister s_indextts2_register;
} // anonymous namespace
