#pragma once

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "frontend/audio_processor.h"
#include "rapidspeech.h" // RS_API
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// X-ASR: Zipformer2 transducer (icefall/k2), zh-en streaming+offline unified.
// One GGUF serves both modes: offline = full-context forward (chunk -1),
// streaming = chunked forward with per-layer left-context caches (M4).
//
// ggml layout convention in this file: sequence tensors are [C, T] with
// ne0 = channels (innermost), batch = 1 always.

struct XASRHParams {
  int num_stacks = 6;
  std::vector<int> num_layers;     // per stack, e.g. 2,2,4,5,4,2
  std::vector<int> downsampling;   // 1,2,4,8,4,2
  std::vector<int> encoder_dim;    // 192,256,512,768,512,256
  std::vector<int> ff_dim;         // 512,768,1536,2048,1536,768
  std::vector<int> num_heads;      // 4,4,4,8,4,4
  std::vector<int> cnn_kernel;     // 31,31,15,15,15,31
  std::vector<int> query_head_dim; // 32 x6
  std::vector<int> pos_head_dim;   // 4 x6
  std::vector<int> value_head_dim; // 12 x6
  int pos_dim = 48;
  int feature_dim = 80;
  int decoder_dim = 512;
  int joiner_dim = 512;
  int context_size = 2;
  int vocab_size = 5000;
  int blank_id = 0;
  int left_context_frames = 96; // at 50 Hz, per-stack = 96 / ds
  int pad_length = 13;          // streaming: T_chunk = decode_chunk_len + 13
};

// Zipformer2EncoderLayer weights (all ggml tensors live on the backend).
struct XASRLayer {
  // RelPositionMultiheadAttentionWeights (shared by attn1/attn2/nonlin)
  struct ggml_tensor *attn_in_proj_w, *attn_in_proj_b; // [(2q+p)h, d]
  struct ggml_tensor *linear_pos_w;                    // [p*h, pos_dim]
  // SelfAttention x2
  struct ggml_tensor *attn1_in_w, *attn1_in_b, *attn1_out_w, *attn1_out_b;
  struct ggml_tensor *attn2_in_w, *attn2_in_b, *attn2_out_w, *attn2_out_b;
  // FeedforwardModule x3 (out_proj = SwooshL + Linear)
  struct ggml_tensor *ff1_in_w, *ff1_in_b, *ff1_out_w, *ff1_out_b;
  struct ggml_tensor *ff2_in_w, *ff2_in_b, *ff2_out_w, *ff2_out_b;
  struct ggml_tensor *ff3_in_w, *ff3_in_b, *ff3_out_w, *ff3_out_b;
  // NonlinAttention
  struct ggml_tensor *na_in_w, *na_in_b, *na_out_w, *na_out_b;
  // ConvolutionModule x2 (ChunkCausalDepthwiseConv1d = causal + chunkwise)
  struct XASRConv {
    struct ggml_tensor *in_w, *in_b;         // pointwise -> 2C (GLU)
    struct ggml_tensor *causal_w, *causal_b; // depthwise k=(K+1)/2
    struct ggml_tensor *chunk_w, *chunk_b;   // depthwise k=K (SAME)
    struct ggml_tensor *out_w, *out_b;       // SwooshR + Linear
    std::vector<float> edge_scale;           // host [2, C, K] (left, right)
  } conv1, conv2;
  // BiasNorm + bypasses
  struct ggml_tensor *norm_bias;
  float norm_scale = 1.0f; // exp(log_scale), host scalar
  struct ggml_tensor *bypass_scale, *bypass_mid_scale; // [d]
};

struct XASRStack {
  std::vector<XASRLayer> layers;
  struct ggml_tensor *downsample_w = nullptr;   // [ds] softmaxed weights
  struct ggml_tensor *out_combiner_w = nullptr; // [d] bypass scale
};

struct XASREmbed {
  struct ggml_tensor *conv0_w, *conv0_b; // Conv2d 1->8   k3 pad(t0,f1)
  struct ggml_tensor *conv4_w, *conv4_b; // Conv2d 8->32  k3 s2
  struct ggml_tensor *conv7_w, *conv7_b; // Conv2d 32->128 k3 s(t1,f2)
  struct ggml_tensor *cnx_dw_w, *cnx_dw_b;   // ConvNeXt depthwise 7x7
  struct ggml_tensor *cnx_pw1_w, *cnx_pw1_b; // 128->384
  struct ggml_tensor *cnx_pw2_w, *cnx_pw2_b; // 384->128
  struct ggml_tensor *out_w, *out_b;         // Linear 2432 -> dim0
  struct ggml_tensor *out_norm_bias;         // BiasNorm
  float out_norm_scale = 1.0f;
  int layer1_channels = 8, layer2_channels = 32, layer3_channels = 128;
};

struct XASRState : public RSState {
  // offline: joiner-projected encoder output, host-side [T25, 512] row-major
  std::vector<float> enc_proj; // [n_frames * joiner_dim]
  int n_enc_frames = 0;
  std::vector<int> tokens; // emitted token ids (greedy)

  // ---- streaming ----
  // per-layer host caches, flattened over stacks in order; ggml layouts:
  //   key   [key_dim, left_s]      nonlin [hc, left_s]
  //   val1/2 [val_dim, left_s]     conv1/2 [k/2, d]  ([T, C] conv layout)
  struct LayerCache {
    std::vector<float> key, nonlin, val1, val2, conv1, conv2;
  };
  std::vector<LayerCache> caches;              // one per encoder layer (19)
  std::vector<float> embed_left_pad;           // [F, 3, C] ggml layout
  int processed_lens = 0;                      // 50 Hz frames seen so far
  std::vector<float> pcm_buffer;               // all pushed PCM
  int fbank_frames_done = 0;                   // frames consumed by chunks
  std::vector<float> fbank_stream;             // rolling fbank cache
  int fbank_stream_start = 0;                  // first frame index in cache
  // transducer continuation
  std::vector<int> ctx_ids;
  std::vector<float> dec_out;
  bool stream_initialized = false;
  // profiling accumulators (reported by FinishStream at INFO level)
  double t_fbank = 0, t_encode = 0, t_decode = 0;
  int n_chunks = 0;

  ~XASRState() override = default;
};

class RS_API XASRModel : public ISpeechModel {
public:
  XASRModel();
  ~XASRModel() override;

  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override;
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;
  std::string GetTranscription(RSState &state) override;
  const RSModelMeta &GetMeta() const override { return meta_; }

  // ---- true streaming API (driven directly by rs-asr-online, not via
  // RSProcessor::Process) ----
  // Set the streaming chunk length in fbank frames (16/48/96/192 ~
  // 160/480/960/1920 ms). Must be called before the first chunk.
  void SetChunkLen(int decode_chunk_len) { chunk_len_ = decode_chunk_len; }
  int GetChunkLen() const { return chunk_len_; }
  // Push PCM ([-1,1] @16k); processes all complete chunks, extending
  // state.tokens. Returns false on error.
  bool EncodeStreamingChunk(const float *pcm, size_t n_samples,
                            XASRState &state, ggml_backend_sched_t sched);
  // Flush: pad with silence and process the tail.
  bool FinishStream(XASRState &state, ggml_backend_sched_t sched);

  int total_layers() const {
    int n = 0;
    for (int x : hp_.num_layers) n += x;
    return n;
  }

private:
  friend struct XASRGraphBuilder;

  RSModelMeta meta_;
  XASRHParams hp_;
  std::unordered_map<int, std::string> id_to_token_;

  XASREmbed embed_;
  std::vector<XASRStack> stacks_;
  struct ggml_tensor *downsample_output_w = nullptr; // [2]
  struct ggml_tensor *joiner_enc_proj_w = nullptr, *joiner_enc_proj_b = nullptr;
  struct ggml_tensor *joiner_dec_proj_w = nullptr, *joiner_dec_proj_b = nullptr;
  struct ggml_tensor *joiner_out_w = nullptr, *joiner_out_b = nullptr;

  // stateless predictor, host-side (tiny: embedding lookup + grouped conv k=2)
  std::vector<float> dec_embed_;  // [vocab, decoder_dim]
  std::vector<float> dec_conv_w_; // [decoder_dim, in_per_group, context_size]
  int dec_conv_in_per_group_ = 4;

  std::unique_ptr<AudioProcessor> audio_proc_;
  ggml_backend_t backend_ = nullptr;

  // dedicated schedulers so streaming encode and greedy joiner graphs never
  // reset each other or the context's main scheduler
  std::vector<ggml_backend_t> backends_;
  ggml_backend_sched_t enc_sched_ = nullptr;
  ggml_backend_sched_t dec_sched_ = nullptr;

  // scratch graph context
  struct ggml_context *ctx_ = nullptr;
  struct ggml_cgraph *gf_ = nullptr;

  bool MapTensors(std::map<std::string, struct ggml_tensor *> &tensors);
  float ReadScalar(struct ggml_tensor *t);
  void ReadHostTensor(struct ggml_tensor *t, std::vector<float> &out);

  // host predictor: context ids -> decoder_out [decoder_dim]
  void RunPredictor(const int *ctx_ids, std::vector<float> &decoder_out) const;

  // offline encoder: fbank [T,80] row-major -> enc_proj on state
  bool EncodeOffline(const std::vector<float> &fbank, int n_frames,
                     XASRState &st, ggml_backend_sched_t sched);

  // streaming: run one chunk (fbank [T_chunk, 80]) through the cached
  // encoder; appends enc_proj rows to the state. Defined in
  // xasr_zipformer.cpp.
  bool EncodeStreamChunkGraph(const std::vector<float> &fbank_chunk,
                              XASRState &st, ggml_backend_sched_t sched);
  void InitStreamState(XASRState &st);
  // greedy over enc_proj rows [from_frame, n_enc_frames), continuing the
  // state's transducer context. Defined in xasr.cpp.
  bool GreedyDecodeRange(XASRState &st, int from_frame,
                         ggml_backend_sched_t sched);

  int chunk_len_ = 32; // decode_chunk_len in fbank frames (default 320 ms)
};

void rs_register_xasr();
