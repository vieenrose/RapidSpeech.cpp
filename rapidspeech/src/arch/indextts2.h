#pragma once

// IndexTTS-2 scaffold.
//
// Pipeline (target, mirrors upstream indextts/infer_v2.py):
//
//   text → BPE → UnifiedVoice GPT-2 (24L, 1280d, 20h, mel_vocab=8194)
//        ↘                                      ↘
//          ConformerEnc(6L, 8h, 512d) + Perceiver(32 latents, 1280d)  ←─ spk_cond_emb
//          ConformerEnc(4L, 4h, 512d) + Perceiver(1  latent,  1024d)  ←─ emo_cond_emb
//          merge_emovec → AR (mel codes ∈ [0,8194), stop=8193)
//          ↓
//          AR forward again → hidden_states  (lm_latent)
//          ↓
//   S2Mel: gpt_layer(1280→256→128→1024) → length_regulator + S_ref
//        → DiT(13L, 512d, 8h) + WaveNet(8L, K=5)  → mel(80, 22.05kHz)
//          ↓
//   BigVGAN-v2 (initial_channel=1536, upsample [4,4,2,2,2,2]) → 22.05kHz wav
//
// Voice cloning uses CAMPPlus (xvec, 192d) on the ref wav.
// Semantic codec (MaskGCT vocos, vocab=8192, codebook_dim=8) is used to
// quantise the w2v-bert hidden states into S_ref / vq2emb(codes) → S_infer.
//
// This scaffold file declares the **weight pointers** and **public model
// class**. Forward graphs live in subsequent CLs (indextts2_gpt.cpp,
// indextts2_s2mel.cpp, indextts2_bigvgan.cpp).

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declared in global namespace; defined in arch/campplus.h.
class CAMPPlusModel;

// Forward-declared; defined in arch/llm_model.h (Qwen3 emotion classifier).
class llm_model;

// Forward-declared; defined in arch/w2v_bert.h.
namespace w2v_bert { class W2VBertModel; }
namespace semantic_codec { class SemanticCodecModel; }

namespace indextts2 {

// -----------------------------------------------------------------------------
// Hyperparameters (mirror config.yaml; populated from GGUF metadata in Load)
// -----------------------------------------------------------------------------
struct HParams {
    // ---- GPT-2 AR ----
    int n_layer       = 24;
    int n_embd        = 1280;
    int n_head        = 20;
    int head_dim      = 64;     // = n_embd / n_head
    int ff_dim        = 5120;   // 4 * n_embd
    int text_vocab    = 12000;  // BPE size
    int mel_codes     = 8194;   // 8192 codebook + start + stop
    int start_mel     = 8192;
    int stop_mel      = 8193;
    int max_text_tok  = 600;
    int max_mel_tok   = 1815;

    // ---- Conditioning encoders (Conformer + Perceiver) ----
    int cond_blocks   = 6;
    int cond_heads    = 8;
    int cond_dim      = 512;
    int cond_ff       = 2048;
    int cond_latents  = 32;     // perceiver num_latents (speaker path)

    int emo_blocks    = 4;
    int emo_heads     = 4;
    int emo_ff        = 1024;
    int emo_latents   = 1;      // perceiver num_latents (emo path)
    int emo_dim       = 1024;   // perceiver inner dim for emo path

    // ---- S2Mel (flow-matching DiT + WaveNet head) ----
    int s2mel_sr            = 22050;
    int s2mel_n_fft         = 1024;
    int s2mel_hop           = 256;
    int s2mel_win           = 1024;
    int s2mel_n_mels        = 80;
    int s2mel_dit_depth     = 13;
    int s2mel_dit_hidden    = 512;
    int s2mel_dit_heads     = 8;
    int s2mel_wavenet_layers = 8;
    int s2mel_wavenet_kernel = 5;
    int s2mel_lr_channels    = 512;
    int s2mel_lr_in_channels = 1024;
    int s2mel_style_dim      = 192;

    // ---- Semantic codec ----
    int sc_vocab        = 8192;
    int sc_hidden       = 1024;
    int sc_codebook_dim = 8;

    // ---- Emotion classifier head sizes (8 groups) ----
    std::vector<int> emo_num = { 3, 17, 2, 8, 4, 5, 10, 24 };

    // ---- BigVGAN-v2 ----
    int bigvgan_sr              = 22050;
    int bigvgan_initial_channel = 1536;
    int bigvgan_n_mels          = 80;
    std::vector<int> bigvgan_upsample_rates        = { 4, 4, 2, 2, 2, 2 };
    std::vector<int> bigvgan_upsample_kernel_sizes = { 8, 8, 4, 4, 4, 4 };
    std::vector<int> bigvgan_resblock_kernel_sizes = { 3, 7, 11 };
    std::vector<int> bigvgan_resblock_dilations    = { 1, 3, 5, 1, 3, 5, 1, 3, 5 };
    bool bigvgan_snake_logscale    = true;
    bool bigvgan_use_tanh_at_final = false;
    bool bigvgan_use_bias_at_final = false;
};

// -----------------------------------------------------------------------------
// GPT-2 block (HuggingFace GPT2 layout; c_attn/c_proj are Conv1D[in,out])
// -----------------------------------------------------------------------------
struct GPT2Block {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *attn_qkv_w = nullptr, *attn_qkv_b = nullptr;   // c_attn
    ggml_tensor *attn_proj_w = nullptr, *attn_proj_b = nullptr; // c_proj
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *mlp_fc_w = nullptr, *mlp_fc_b = nullptr;       // c_fc
    ggml_tensor *mlp_proj_w = nullptr, *mlp_proj_b = nullptr;   // c_proj
};

// -----------------------------------------------------------------------------
// Conformer block (speaker / emo conditioning encoders). Held as opaque tensor
// pointers indexed by raw GGUF subkey — the forward will look them up.
// -----------------------------------------------------------------------------
struct ConformerBlock {
    // ffn_macaron_*, self_attn_*, conv_*, ffn_*, norm_* — populated from
    // tensor name suffixes during Load.
    std::unordered_map<std::string, ggml_tensor *> by_name;
};

struct ConformerEncoder {
    int n_blocks = 0;
    std::vector<ConformerBlock> blocks;
    // Pre/post tensors (subsampler, output norm, etc.)
    std::unordered_map<std::string, ggml_tensor *> by_name;
};

// -----------------------------------------------------------------------------
// Perceiver resampler (lucidrains-style; num_latents queries cross-attend ctx)
// -----------------------------------------------------------------------------
struct PerceiverLayer {
    // attn: to_q, to_kv, to_out; ff: 2x linear (GEGLU)
    std::unordered_map<std::string, ggml_tensor *> by_name;
};

struct PerceiverResampler {
    ggml_tensor *latents     = nullptr;
    ggml_tensor *proj_ctx_w  = nullptr;  // proj_context (Identity when dim_ctx==dim)
    ggml_tensor *proj_ctx_b  = nullptr;
    std::vector<PerceiverLayer> layers;
    ggml_tensor *norm_gamma  = nullptr;  // final RMSNorm
};

// -----------------------------------------------------------------------------
// S2Mel sub-modules (DiT estimator + WaveNet head + length_regulator +
// gpt_layer MLP). Tensors are again held by name; forward graph in the next CL.
// -----------------------------------------------------------------------------
struct S2MelWeights {
    // models.gpt_layer.0/1/2 (1280→256→128→1024)
    ggml_tensor *gpt_layer0_w = nullptr, *gpt_layer0_b = nullptr;
    ggml_tensor *gpt_layer1_w = nullptr, *gpt_layer1_b = nullptr;
    ggml_tensor *gpt_layer2_w = nullptr, *gpt_layer2_b = nullptr;

    // models.length_regulator (InterpolateRegulator: in 1024 → 512 ch)
    std::unordered_map<std::string, ggml_tensor *> length_regulator;

    // models.cfm.estimator.* (DiT depth=13 + WaveNet head)
    std::unordered_map<std::string, ggml_tensor *> cfm;
};

// -----------------------------------------------------------------------------
// Semantic codec (MaskGCT vocos-style 12L encoder, 8192-entry codebook)
// -----------------------------------------------------------------------------
struct SemanticCodecWeights {
    // encoder.*, quantizer.codebook / in_proj / out_proj, decoder.*
    std::unordered_map<std::string, ggml_tensor *> tensors;
};

// -----------------------------------------------------------------------------
// CAMPPlus speaker encoder: we instantiate the standalone CAMPPlusModel from
// arch/campplus.cpp directly, using its `tensor_prefix` parameter to read
// tensors from the IndexTTS-2 GGUF under the `indextts2.campplus.` namespace.
// (CAMPPlusModel is forward-declared in the global namespace above.)
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// BigVGAN-v2 (loaded from the separate indextts2-bigvgan.gguf at runtime)
// -----------------------------------------------------------------------------
struct BigVGANWeights {
    std::unordered_map<std::string, ggml_tensor *> tensors;

    // Lifetime-owning members for the external GGUF.
    gguf_context        *ctx_gguf = nullptr;
    ggml_context        *ctx_data = nullptr;
    ggml_backend_buffer_t buf     = nullptr;
};

// -----------------------------------------------------------------------------
// SentencePiece-unigram BPE tokenizer (Viterbi). Vocab + scores are shipped in
// the GGUF as tokenizer.ggml.tokens / tokenizer.ggml.scores so we don't need
// to link against libsentencepiece at runtime.
// -----------------------------------------------------------------------------
struct BPETokenizer {
    std::vector<std::string>                 id_to_token;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::vector<float>                       scores;
    bool                                     loaded = false;

    bool                 load_from_vocab(std::vector<std::string> tokens,
                                         std::vector<float> scores_in);
    std::vector<int32_t> encode(const std::string &text) const;
};

// -----------------------------------------------------------------------------
// Per-request state
// -----------------------------------------------------------------------------
struct State : public RSState {
    // Inputs.
    std::string text_in;
    std::string language;
    std::string instruct;
    std::vector<int32_t> text_token_ids;
    std::vector<int32_t> emo_text_token_ids;
    std::vector<float>   ref_audio;     // 16k mono, ref for w2v-bert + semantic codec
    int                  ref_sr = 16000;
    std::vector<float>   ref_audio_22k; // resampled copy fed to S2Mel mel_ref
    std::vector<float>   ref_audio_24k; // ECAPA path (unused; CAMPPlus is 16k)
    std::vector<float>   spk_embedding; // [192] CAMPPlus output
    std::vector<float>   emo_vec;       // 8-way prob vector

    // W2V-BERT + semantic_codec hidden states for the AR conditioning path
    // (Conformer + PerceiverResampler). Layout: [T_sc * sc_hidden] f32,
    // ne[0]=sc_hidden fastest (== PT [T_sc, sc_hidden] C-fast). Populated by
    // PushReferenceAudio once the W2V-BERT + semantic_codec stages land
    // (task #10) — or by the env override RS_INDEXTTS2_SC_HIDDEN_NPY for
    // bring-up. Left empty → AR runs with zero spk_latents / emo_vec.
    std::vector<float>   sc_hidden;
    int                  T_sc = 0;

    // ---- Emotion control (mirrors infer_v2.py infer() emotion args) ----
    // emo_mode: 0=from speaker (follow voice), 1=from external emo audio,
    //           2=from 8-d emotion vector, 3=from emo text (QwenEmotion).
    int                  emo_mode      = 0;
    float                emo_alpha     = 1.0f;  // emo_weight; webui default 0.65
    bool                 emo_use_random = false;
    bool                 emo_apply_bias = false; // mode 2: normalize_emo_vec bias
    std::vector<float>   emo_vector;             // [8] happy..calm; empty if unused
    std::string          emo_text;               // mode 3 text; empty → use main text
    // Independent emo reference audio → its own sc_hidden (same pipeline as the
    // speaker path). Empty → emo conditioning falls back to the speaker audio.
    std::vector<float>   emo_sc_hidden;
    int                  emo_T_sc = 0;

    // Intermediates.
    std::vector<int32_t> mel_codes;     // AR output (≤ max_mel_tok)
    std::vector<float>   lm_latent;     // GPT hidden states [T_mel, n_embd]
    std::vector<float>   mel_pred;      // S2Mel output [T_mel_22k, n_mels]

    // Outputs.
    std::vector<float>   audio_output;  // 22.05kHz f32

    // Sampling.
    std::mt19937 rng{ 0xC05A3 };
    float temperature = 0.8f;
    float top_p       = 0.8f;
    int   top_k       = 30;
};

// -----------------------------------------------------------------------------
// Public model class
// -----------------------------------------------------------------------------
class Model : public ISpeechModel {
public:
    Model();
    ~Model() override;

    // ISpeechModel
    bool Load(const std::unique_ptr<rs_context_t> &ctx,
              ggml_backend_t backend) override;
    std::shared_ptr<RSState> CreateState() override;

    bool Encode(const std::vector<float> &input_frames, RSState &state,
                ggml_backend_sched_t sched) override;
    bool Decode(RSState &state, ggml_backend_sched_t sched) override;

    std::string GetTranscription(RSState &state) override { (void)state; return ""; }
    const RSModelMeta &GetMeta() const override { return meta_; }

    // TTS interface
    bool PushText(RSState &state, const char *text,
                  const char *language = nullptr,
                  const char *instruct = nullptr) override;
    bool PushReferenceAudio(RSState &state, const float *samples, int n_samples,
                            int sample_rate, ggml_backend_sched_t sched) override;
    bool PushReferenceText(RSState &state, const char *ref_text) override;
    int  GetAudioOutput(RSState &state, float **out_data) override;
    void SetSeed(uint64_t seed) override { seed_ = seed; }

    // Independent emotion control (mirrors infer_v2.py emotion args).
    bool PushEmotionAudio(RSState &state, const float *samples, int n_samples,
                          int sample_rate, ggml_backend_sched_t sched) override;
    void SetEmotionControl(RSState &state, int mode, float emo_alpha,
                           const float *vec8, bool use_random, bool apply_bias,
                           const char *emo_text) override;

public:
    // GPT-2 KV-cache scratch context. Public so the free helper
    // `build_gpt_step_graph` in indextts2_gpt.cpp can name it; the type is an
    // implementation detail of the AR forward path.
    struct GPTContext;

private:
    bool LoadGPT(struct ggml_context *gguf_data);
    bool LoadS2Mel(struct ggml_context *gguf_data);
    bool LoadSemanticCodec(struct ggml_context *gguf_data, gguf_context *ctx_gguf,
                           ggml_backend_t backend);
    bool LoadCampplus(struct ggml_context *gguf_data, gguf_context *ctx_gguf,
                      ggml_backend_t backend);
    bool LoadW2VBert(struct ggml_context *gguf_data, gguf_context *ctx_gguf,
                     ggml_backend_t backend);
    bool LoadStaticMatrices(struct ggml_context *gguf_data);
    bool LoadHParamsFromGGUF(gguf_context *ctx_gguf);

    // BigVGAN GGUF is shipped separately — loaded lazily on first Decode using
    // either a CLI-provided path or the env var RS_INDEXTTS2_BIGVGAN_GGUF.
    bool LoadBigVGAN(const std::string &path, ggml_backend_t backend);

    // Optional Qwen3 emotion model (qwen0.6bemo4-merge), also separate GGUF.
    bool LoadQwenEmotion(const std::string &path, ggml_backend_t backend);

    // QwenEmoInfer: run the Qwen3 emotion classifier on `text`, returning the
    // 8-d emotion vector [happy, angry, sad, afraid, disgusted, melancholic,
    // surprised, calm]. Mirrors infer_v2.py QwenEmotion.inference(). Defined in
    // indextts2_qwen_emo.cpp. Returns false if the model is unavailable.
    bool QwenEmoInfer(ggml_backend_sched_t sched, const std::string &text,
                      std::vector<float> &vec8);

    // ComputeScHidden: fbank → W2V-BERT → per-channel normalize → semantic_codec
    // quantize, producing the [T*sc_hidden] feature buffer the conditioning
    // encoders consume. `pcm16` must be 16 kHz mono. Shared by PushReferenceAudio
    // and PushEmotionAudio. Returns false on failure (out left empty).
    bool ComputeScHidden(const float *pcm16, int n16,
                         std::vector<float> &out, int &T_out,
                         ggml_backend_sched_t sched, bool is_emo);

    // ResolveEmotion: assembles the final emo conditioning vector (`emovec`,
    // [n_embd]) per the active State::emo_mode, mirroring infer_v2.py
    // infer_generator emotion handling (base/alpha blend + emo_matrix vector
    // path + QwenEmotion). Defined in indextts2.cpp.
    bool ResolveEmotion(State &s, ggml_backend_sched_t sched,
                        std::vector<float> &emovec);

    // ComputeEmovecMat: emo_matrix vector path (per-group prototype select via
    // CAMPPlus style cosine, weighted sum). Output: [n_embd].
    bool ComputeEmovecMat(State &s, const std::vector<float> &wvec,
                          std::vector<float> &out);

    // normalize_emo_vec: optional bias de-emphasis + sum≤0.8 cap (mode 2).
    static std::vector<float> normalize_emo_vec(const std::vector<float> &vec8,
                                                bool apply_bias);

    // ---- AR forward path (indextts2_gpt.cpp) ----------------------------
    // RunConditioning: runs Conformer(spk) + Perceiver(spk) over the semantic
    // codec hidden states to produce 32 conditioning latent vectors of size
    // n_embd (1280). The result is written into `cond_latents`.
    bool RunConditioning(State &s, ggml_backend_sched_t sched,
                         const std::vector<float> &sc_hidden, int T_sc,
                         std::vector<float> &cond_latents);

    // RunGetEmovec: runs the smaller (4L / 4h) Conformer + Perceiver (1 latent)
    // over the emo-side semantic hidden states, then emovec_layer (1024→1280)
    // and emo_layer (1280→1280). Mirrors upstream gpt.get_emovec(). Output: a
    // single n_embd (1280) vector — the full emo conditioning embedding.
    bool RunGetEmovec(State &s, ggml_backend_sched_t sched,
                      const std::vector<float> &emo_hidden, int T_emo,
                      std::vector<float> &emo_latent);

    // RunAR: drives the GPT-2 autoregressive loop — builds prefix from
    // {cond_latents, text_tokens, start_mel}, samples mel codes one-by-one
    // until stop_mel, then performs a second full-context forward to extract
    // hidden states (lm_latent) for the S2Mel stage.
    bool RunAR(State &s, ggml_backend_sched_t sched);

    // GPT-2 KV-cached single-token step. Used by RunAR.
    bool GPTPrefill(GPTContext &gc, ggml_backend_sched_t sched,
                    const std::vector<float> &embeds, int n_tokens);
    int  GPTSampleStep(GPTContext &gc, ggml_backend_sched_t sched,
                       const std::vector<float> &embed, int mel_pos,
                       State &s);
    bool GPTLatentPass(GPTContext &gc, ggml_backend_sched_t sched,
                       State &s, std::vector<float> &lm_latent_out);

    // ---- S2Mel forward path (indextts2_s2mel.cpp) -----------------------
    // RunS2Mel: gpt_layer(lm_latent) → length_regulator → CFM Euler solver
    // over the DiT estimator → mel-spectrogram [T_mel, n_mels].
    bool RunS2Mel(State &s, ggml_backend_sched_t sched);

    // Smoke harness for S2Mel: when env RS_INDEXTTS2_S2MEL_TEST_DIR is set,
    // loads lm_latent.npy etc. from that dir, runs gpt_layer +
    // length_regulator, and dumps gpt_layer_out.f32 / lr_out.f32 next to the
    // .npy references. Sets `taken=true` if it handled this decode call.
    bool RunS2MelSmoke(State &s, ggml_backend_sched_t sched, bool &taken);

    // Smoke harness for GPT-2 prefill bit-exact check. When env
    // RS_INDEXTTS2_GPT_TEST_DIR is set, loads inputs_embeds.npy from that
    // dir, runs one prefill pass through build_gpt_step_graph (skipping
    // build_prefix_embeds / Conformer / Perceiver / emo entirely), and
    // dumps gpt_last_hidden.f32 / final_norm_out.f32 / mel_head_logits.f32
    // plus first_sampled_token.txt for scripts/diff_gpt_prefill.py.
    bool RunGPTPrefillSmoke(State &s, ggml_backend_sched_t sched, bool &taken);

    // Smoke harness for W2V-BERT 2.0. Reads input_features.npy from
    // RS_INDEXTTS2_W2V_BERT_TEST_DIR, runs forward through all 18 Conformer
    // layers, and dumps w2v_hidden0.f32 / w2v_hidden17.f32.
    bool RunW2VBertSmoke(State &s, ggml_backend_sched_t sched, bool &taken);

    // ---- BigVGAN-v2 vocoder (indextts2_bigvgan.cpp) ---------------------
    // RunBigVGAN: mel [n_mels, T] → 22.05kHz f32 waveform [T*256]. Builds a
    // one-shot graph against the lazily-loaded bigvgan_ context, computes
    // through `backend_`, and writes State::audio_output.
    //
    // `mel` is the C++ end of the S2Mel output. The smoke driver feeds the
    // PyTorch reference dump (scripts/dump_bigvgan_ref.py / mel.npy) to
    // validate bit-exactness against wav.npy.
    bool RunBigVGAN(State &s, ggml_backend_sched_t sched,
                    const float *mel, int n_mels, int T_mel);

private:
    RSModelMeta meta_;
    HParams     hp_;
    uint64_t    seed_ = 0xC05A3ULL;

    // ---- Tokenizer / front-end ----
    BPETokenizer bpe_;

    // ---- GPT-2 AR ----
    ggml_tensor *text_emb_w_      = nullptr;
    ggml_tensor *mel_emb_w_       = nullptr;
    ggml_tensor *text_pos_emb_w_  = nullptr;
    ggml_tensor *mel_pos_emb_w_   = nullptr;
    ggml_tensor *gpt_wpe_w_       = nullptr;
    std::vector<GPT2Block> blocks_;
    ggml_tensor *gpt_ln_f_w_ = nullptr, *gpt_ln_f_b_ = nullptr;
    ggml_tensor *final_norm_w_ = nullptr, *final_norm_b_ = nullptr;
    ggml_tensor *mel_head_w_   = nullptr, *mel_head_b_   = nullptr;
    ggml_tensor *text_head_w_  = nullptr, *text_head_b_  = nullptr;

    // ---- Conditioning ----
    ConformerEncoder    cond_enc_;       // 6-layer
    PerceiverResampler  cond_perceiver_; // 32 latents
    ConformerEncoder    emo_cond_enc_;       // 4-layer
    PerceiverResampler  emo_cond_perceiver_; // 1 latent
    ggml_tensor *emovec_layer_w_ = nullptr, *emovec_layer_b_ = nullptr;
    ggml_tensor *emo_layer_w_    = nullptr, *emo_layer_b_    = nullptr;
    ggml_tensor *speed_emb_w_    = nullptr;

    // ---- S2Mel + Semantic codec + CAMPPlus ----
    S2MelWeights         s2mel_;
    SemanticCodecWeights sc_;
    std::shared_ptr<::CAMPPlusModel> campplus_model_;

    // ---- W2V-BERT 2.0 (loaded from indextts2.w2v_bert.* in the same GGUF) ----
    std::shared_ptr<::w2v_bert::W2VBertModel> w2v_bert_model_;
    std::shared_ptr<::semantic_codec::SemanticCodecModel> semantic_codec_model_;

    // ---- Qwen3 emotion classifier (separate GGUF, lazily loaded) ----
    std::shared_ptr<llm_model> qwen_emo_model_;
    bool qwen_emo_ready_ = false;

    // ---- Static matrices ----
    ggml_tensor *spk_matrix_ = nullptr;  // feat1.pt
    ggml_tensor *emo_matrix_ = nullptr;  // feat2.pt
    ggml_tensor *w2v_mean_   = nullptr;
    ggml_tensor *w2v_std_    = nullptr;

    // ---- BigVGAN (loaded lazily) ----
    BigVGANWeights bigvgan_;
    bool           bigvgan_ready_ = false;
#ifdef RS_USE_METAL_BIGVGAN
    std::unique_ptr<class BigVGANMetalDecoder> bigvgan_metal_;
#endif

    // ---- Backends ----
    ggml_backend_t backend_     = nullptr;
    ggml_backend_t cpu_backend_ = nullptr;
};

} // namespace indextts2
