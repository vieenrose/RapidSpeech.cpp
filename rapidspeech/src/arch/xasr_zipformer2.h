#pragma once

// X-ASR zh-en streaming **zipformer2 transducer** (k2-fsa / sherpa-onnx export),
// ported to ggml for RapidSpeech.cpp. GGUF arch id: "XAsrZipformer2".
//
// Produced by tools/convert_xasr_to_gguf.py (in the x-asr-rapidspeech repo).
// Architecture: encoder_embed (Conv2dSubsampling + ConvNeXt + BiasNorm) ->
// 6 zipformer2 stacks (19 layers total, with downsample/upsample + out_combiner)
// -> encoder_proj -> encoder_out[512]; stateless decoder (embedding + Conv1d,
// context_size=2) + joiner (Add->Tanh->Linear) -> 5000 logits. Streaming via
// per-layer caches (key / nonlin_attn / val1 / val2 / conv1 / conv2) + embed
// state, mirroring the ONNX state contract.

#include "core/rs_context.h"
#include "core/rs_model.h"
#include "rapidspeech.h"
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

struct XAsrHParams {
  int n_stacks = 0;
  std::vector<int> num_layers;     // per stack
  std::vector<int> dims;           // per stack (embed_dim)
  std::vector<int> cnn_kernels;    // per stack
  std::vector<int> left_context;   // per stack
  std::vector<int> query_head_dim; // per stack
  std::vector<int> value_head_dim; // per stack
  std::vector<int> num_heads;      // per stack
  int decode_chunk_len = 96;
  int T = 109;
  int feat_dim = 80;
  int out_dim = 512;   // encoder_out / joiner_dim
  int joiner_dim = 512;
  int context_size = 2;
  int vocab_size = 5000;
  int sample_rate = 16000;
};

// Per-request streaming state: the recurrent caches + the running hypothesis.
struct XAsrState : public RSState {
  // Encoder streaming caches, one group per layer (flattened across stacks),
  // held as host buffers and uploaded as graph inputs each chunk. Filled in
  // during the streaming-state stage.
  std::vector<std::vector<float>> cached_key, cached_nonlin_attn,
      cached_val1, cached_val2, cached_conv1, cached_conv2;
  std::vector<float> embed_states;
  int64_t processed_lens = 0;

  // Transducer hypothesis (token ids); first context_size entries are blanks.
  std::vector<int32_t> hyp;
  // Latest encoder_out for the current chunk: [Tc, out_dim] row-major.
  std::vector<float> encoder_out;
  int enc_T = 0;
  bool initialized = false;
};

class RS_API XAsrZipformer2Model : public ISpeechModel {
public:
  XAsrZipformer2Model();
  ~XAsrZipformer2Model() override;

  bool Load(const std::unique_ptr<rs_context_t> &ctx,
            ggml_backend_t backend) override;
  std::shared_ptr<RSState> CreateState() override;
  bool Encode(const std::vector<float> &input_frames, RSState &state,
              ggml_backend_sched_t sched) override;
  bool Decode(RSState &state, ggml_backend_sched_t sched) override;
  std::string GetTranscription(RSState &state) override;
  const RSModelMeta &GetMeta() const override { return meta_; }

  int GetBlankId() const { return 0; }
  const std::unordered_map<int, std::string> &GetIdToToken() const {
    return id_to_token_;
  }

private:
  RSModelMeta meta_;
  XAsrHParams hp_;
  std::unordered_map<int, std::string> id_to_token_;
  // All GGUF weights, indexed by their canonical name (see converter).
  std::map<std::string, struct ggml_tensor *> w_;

  struct ggml_tensor *W(const std::string &name) const;
};

// --- dev/parity entrypoints (built by the xasr-dev-test target) ---
// Run only encoder_embed on one chunk of features [T, feat_dim] (row-major,
// freq fastest) with a zero left-pad cache, returning [T_out * out_dim] row
// major. Used to numerically validate against scripts/xasr/dump_reference.py.
RS_API bool xasr_debug_embed(const std::map<std::string, struct ggml_tensor *> &w,
                             ggml_backend_t backend, const float *feats, int T,
                             int feat_dim, std::vector<float> &out, int *T_out,
                             int *dim_out);
