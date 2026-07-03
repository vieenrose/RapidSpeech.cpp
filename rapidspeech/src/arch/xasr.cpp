// X-ASR (Zipformer2 transducer) model: loading, transducer greedy decoding,
// detokenization, arch registration. Encoder graph: xasr_zipformer.cpp.

#include "arch/xasr.h"
#include "ggml.h"
#include "utils/rs_log.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static int kv_i32(gguf_context *g, const char *key) {
  const int i = gguf_find_key(g, key);
  if (i < 0) {
    RS_LOG_ERR("XASR: missing GGUF key %s", key);
    return -1;
  }
  return gguf_get_val_i32(g, i);
}

static bool kv_i32_arr(gguf_context *g, const char *key,
                       std::vector<int> &out) {
  const int i = gguf_find_key(g, key);
  if (i < 0) {
    RS_LOG_ERR("XASR: missing GGUF array %s", key);
    return false;
  }
  const int n = (int)gguf_get_arr_n(g, i);
  const int32_t *data = (const int32_t *)gguf_get_arr_data(g, i);
  out.assign(data, data + n);
  return true;
}

XASRModel::XASRModel() = default;

XASRModel::~XASRModel() {
  if (enc_sched_) {
    ggml_backend_sched_free(enc_sched_);
    enc_sched_ = nullptr;
  }
  if (dec_sched_) {
    ggml_backend_sched_free(dec_sched_);
    dec_sched_ = nullptr;
  }
  if (ctx_) {
    ggml_free(ctx_);
    ctx_ = nullptr;
  }
}

float XASRModel::ReadScalar(struct ggml_tensor *t) {
  float v = 0.0f;
  ggml_backend_tensor_get(t, &v, 0, sizeof(float));
  return v;
}

void XASRModel::ReadHostTensor(struct ggml_tensor *t,
                               std::vector<float> &out) {
  out.resize(ggml_nelements(t));
  ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

bool XASRModel::Load(const std::unique_ptr<rs_context_t> &ctx,
                     ggml_backend_t backend) {
  if (!ctx || !ctx->ctx_gguf || !ctx->gguf_data) {
    RS_LOG_ERR("XASR: invalid context in Load");
    return false;
  }
  backend_ = backend;
  gguf_context *g = ctx->ctx_gguf;

  hp_.num_stacks = kv_i32(g, "xasr.num_stacks");
  if (!kv_i32_arr(g, "xasr.num_layers", hp_.num_layers) ||
      !kv_i32_arr(g, "xasr.downsampling_factor", hp_.downsampling) ||
      !kv_i32_arr(g, "xasr.encoder_dim", hp_.encoder_dim) ||
      !kv_i32_arr(g, "xasr.feedforward_dim", hp_.ff_dim) ||
      !kv_i32_arr(g, "xasr.num_heads", hp_.num_heads) ||
      !kv_i32_arr(g, "xasr.cnn_module_kernel", hp_.cnn_kernel) ||
      !kv_i32_arr(g, "xasr.query_head_dim", hp_.query_head_dim) ||
      !kv_i32_arr(g, "xasr.pos_head_dim", hp_.pos_head_dim) ||
      !kv_i32_arr(g, "xasr.value_head_dim", hp_.value_head_dim)) {
    return false;
  }
  hp_.pos_dim = kv_i32(g, "xasr.pos_dim");
  hp_.feature_dim = kv_i32(g, "xasr.feature_dim");
  hp_.decoder_dim = kv_i32(g, "xasr.decoder_dim");
  hp_.joiner_dim = kv_i32(g, "xasr.joiner_dim");
  hp_.context_size = kv_i32(g, "xasr.context_size");
  hp_.vocab_size = kv_i32(g, "xasr.vocab_size");
  hp_.blank_id = kv_i32(g, "xasr.blank_id");
  hp_.left_context_frames = kv_i32(g, "xasr.left_context_frames");
  hp_.pad_length = kv_i32(g, "xasr.pad_length");

  meta_.arch_name = "x-asr";
  meta_.audio_sample_rate = 16000;
  meta_.n_mels = hp_.feature_dim;
  meta_.vocab_size = hp_.vocab_size;
  // X-ASR owns its frontend: sherpa-onnx style kaldi fbank with
  // snip_edges=false and mel high freq 7600 differs from the shared one.
  meta_.use_external_frontend = true;
  meta_.window_type = WindowType::POVEY;

  const int tok_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
  if (tok_idx >= 0) {
    const int n = (int)gguf_get_arr_n(g, tok_idx);
    for (int i = 0; i < n; i++) {
      id_to_token_[i] = gguf_get_arr_str(g, tok_idx, i);
    }
  }

  STFTConfig cfg;
  cfg.sample_rate = 16000;
  cfg.frame_size = 400;
  cfg.frame_step = 160;
  cfg.n_mels = hp_.feature_dim;
  cfg.window_type = WindowType::POVEY;
  cfg.snip_edges = false;
  cfg.mel_low_hz = 20.0f;
  cfg.mel_high_hz = 7600.0f; // sherpa-onnx high_freq = -400
  cfg.use_lfr = false;
  cfg.use_cmvn = false;
  audio_proc_ = std::make_unique<AudioProcessor>(cfg);

  std::map<std::string, struct ggml_tensor *> tensors;
  const int n_tensors = (int)gguf_get_n_tensors(g);
  for (int i = 0; i < n_tensors; ++i) {
    const char *name = gguf_get_tensor_name(g, i);
    struct ggml_tensor *t = ggml_get_tensor(ctx->gguf_data, name);
    if (t) tensors[name] = t;
  }
  if (!MapTensors(tensors)) return false;

  // Dedicated schedulers (same backend list as the main one; CPU is last as
  // ggml requires). The streaming encoder keeps a persistent allocated graph
  // on enc_sched_; greedy joiner graphs reset dec_sched_ freely. Neither
  // touches the context's main scheduler.
  backends_ = ctx->backends;
  if (!backends_.empty()) {
    enc_sched_ = ggml_backend_sched_new(backends_.data(), nullptr,
                                        (int)backends_.size(), 65536, false,
                                        true);
    dec_sched_ = ggml_backend_sched_new(backends_.data(), nullptr,
                                        (int)backends_.size(), 4096, false,
                                        true);
  }
  return true;
}

bool XASRModel::MapTensors(
    std::map<std::string, struct ggml_tensor *> &tensors) {
  auto get = [&](const std::string &name) -> struct ggml_tensor * {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      throw std::runtime_error("XASR: missing tensor " + name);
    }
    return it->second;
  };

  try {
    // encoder_embed
    embed_.conv0_w = get("encoder_embed.conv.0.weight");
    embed_.conv0_b = get("encoder_embed.conv.0.bias");
    embed_.conv4_w = get("encoder_embed.conv.4.weight");
    embed_.conv4_b = get("encoder_embed.conv.4.bias");
    embed_.conv7_w = get("encoder_embed.conv.7.weight");
    embed_.conv7_b = get("encoder_embed.conv.7.bias");
    embed_.cnx_dw_w = get("encoder_embed.convnext.depthwise_conv.weight");
    embed_.cnx_dw_b = get("encoder_embed.convnext.depthwise_conv.bias");
    embed_.cnx_pw1_w = get("encoder_embed.convnext.pointwise_conv1.weight");
    embed_.cnx_pw1_b = get("encoder_embed.convnext.pointwise_conv1.bias");
    embed_.cnx_pw2_w = get("encoder_embed.convnext.pointwise_conv2.weight");
    embed_.cnx_pw2_b = get("encoder_embed.convnext.pointwise_conv2.bias");
    embed_.out_w = get("encoder_embed.out.weight");
    embed_.out_b = get("encoder_embed.out.bias");
    embed_.out_norm_bias = get("encoder_embed.out_norm.bias");
    embed_.out_norm_scale =
        std::exp(ReadScalar(get("encoder_embed.out_norm.log_scale")));
    embed_.layer1_channels = (int)embed_.conv0_w->ne[3];
    embed_.layer2_channels = (int)embed_.conv4_w->ne[3];
    embed_.layer3_channels = (int)embed_.conv7_w->ne[3];

    stacks_.resize(hp_.num_stacks);
    for (int s = 0; s < hp_.num_stacks; s++) {
      auto &st = stacks_[s];
      const std::string sp = "enc." + std::to_string(s) + ".";
      if (hp_.downsampling[s] > 1) {
        st.downsample_w = get(sp + "downsample.weight");
        st.out_combiner_w = get(sp + "out_combiner.bypass_scale");
      }
      st.layers.resize(hp_.num_layers[s]);
      for (int l = 0; l < hp_.num_layers[s]; l++) {
        auto &L = st.layers[l];
        const std::string p = sp + "layers." + std::to_string(l) + ".";
        L.attn_in_proj_w = get(p + "self_attn_weights.in_proj.weight");
        L.attn_in_proj_b = get(p + "self_attn_weights.in_proj.bias");
        L.linear_pos_w = get(p + "self_attn_weights.linear_pos.weight");
        L.attn1_in_w = get(p + "self_attn1.in_proj.weight");
        L.attn1_in_b = get(p + "self_attn1.in_proj.bias");
        L.attn1_out_w = get(p + "self_attn1.out_proj.weight");
        L.attn1_out_b = get(p + "self_attn1.out_proj.bias");
        L.attn2_in_w = get(p + "self_attn2.in_proj.weight");
        L.attn2_in_b = get(p + "self_attn2.in_proj.bias");
        L.attn2_out_w = get(p + "self_attn2.out_proj.weight");
        L.attn2_out_b = get(p + "self_attn2.out_proj.bias");
        L.ff1_in_w = get(p + "feed_forward1.in_proj.weight");
        L.ff1_in_b = get(p + "feed_forward1.in_proj.bias");
        L.ff1_out_w = get(p + "feed_forward1.out_proj.weight");
        L.ff1_out_b = get(p + "feed_forward1.out_proj.bias");
        L.ff2_in_w = get(p + "feed_forward2.in_proj.weight");
        L.ff2_in_b = get(p + "feed_forward2.in_proj.bias");
        L.ff2_out_w = get(p + "feed_forward2.out_proj.weight");
        L.ff2_out_b = get(p + "feed_forward2.out_proj.bias");
        L.ff3_in_w = get(p + "feed_forward3.in_proj.weight");
        L.ff3_in_b = get(p + "feed_forward3.in_proj.bias");
        L.ff3_out_w = get(p + "feed_forward3.out_proj.weight");
        L.ff3_out_b = get(p + "feed_forward3.out_proj.bias");
        L.na_in_w = get(p + "nonlin_attention.in_proj.weight");
        L.na_in_b = get(p + "nonlin_attention.in_proj.bias");
        L.na_out_w = get(p + "nonlin_attention.out_proj.weight");
        L.na_out_b = get(p + "nonlin_attention.out_proj.bias");
        auto load_conv = [&](const char *mod, XASRLayer::XASRConv &cv) {
          const std::string cp = p + mod + ".";
          cv.in_w = get(cp + "in_proj.weight");
          cv.in_b = get(cp + "in_proj.bias");
          cv.causal_w = get(cp + "depthwise_conv.causal_conv.weight");
          cv.causal_b = get(cp + "depthwise_conv.causal_conv.bias");
          cv.chunk_w = get(cp + "depthwise_conv.chunkwise_conv.weight");
          cv.chunk_b = get(cp + "depthwise_conv.chunkwise_conv.bias");
          cv.out_w = get(cp + "out_proj.weight");
          cv.out_b = get(cp + "out_proj.bias");
          ReadHostTensor(get(cp + "depthwise_conv.chunkwise_conv_scale"),
                         cv.edge_scale);
        };
        load_conv("conv_module1", L.conv1);
        load_conv("conv_module2", L.conv2);
        L.norm_bias = get(p + "norm.bias");
        L.norm_scale = std::exp(ReadScalar(get(p + "norm.log_scale")));
        L.bypass_scale = get(p + "bypass.bypass_scale");
        L.bypass_mid_scale = get(p + "bypass_mid.bypass_scale");
      }
    }

    downsample_output_w = get("enc.downsample_output.weight");
    joiner_enc_proj_w = get("joiner.encoder_proj.weight");
    joiner_enc_proj_b = get("joiner.encoder_proj.bias");
    joiner_dec_proj_w = get("joiner.decoder_proj.weight");
    joiner_dec_proj_b = get("joiner.decoder_proj.bias");
    joiner_out_w = get("joiner.output_linear.weight");
    joiner_out_b = get("joiner.output_linear.bias");

    // host predictor
    ReadHostTensor(get("decoder.embedding.weight"), dec_embed_);
    struct ggml_tensor *dc = get("decoder.conv.weight");
    dec_conv_in_per_group_ = (int)dc->ne[1];
    ReadHostTensor(dc, dec_conv_w_);
  } catch (const std::exception &e) {
    RS_LOG_ERR("%s", e.what());
    return false;
  }
  RS_LOG_INFO("XASR: mapped tensors OK (%d stacks)", hp_.num_stacks);
  return true;
}

std::shared_ptr<RSState> XASRModel::CreateState() {
  return std::make_shared<XASRState>();
}

// ---------------------------------------------------------------------------
// host predictor: decoder(ctx_ids) = relu(grouped_conv(embedding))
// ---------------------------------------------------------------------------

void XASRModel::RunPredictor(const int *ctx_ids,
                             std::vector<float> &decoder_out) const {
  const int D = hp_.decoder_dim;
  const int cs = hp_.context_size; // 2
  const int ipg = dec_conv_in_per_group_;
  decoder_out.assign(D, 0.0f);
  for (int c = 0; c < D; c++) {
    const int gbase = (c / ipg) * ipg;
    float acc = 0.0f;
    for (int j = 0; j < cs; j++) {
      const float *emb = dec_embed_.data() + (size_t)ctx_ids[j] * D;
      const float *w = dec_conv_w_.data() + ((size_t)c * ipg) * cs;
      for (int i = 0; i < ipg; i++) {
        acc += w[i * cs + j] * emb[gbase + i];
      }
    }
    decoder_out[c] = acc > 0.0f ? acc : 0.0f; // F.relu
  }
}

// ---------------------------------------------------------------------------
// Encode / Decode
// ---------------------------------------------------------------------------

bool XASRModel::Encode(const std::vector<float> &input_frames, RSState &state,
                       ggml_backend_sched_t sched) {
  auto &st = static_cast<XASRState &>(state);

  std::vector<float> fbank;
  int n_frames = 0;
  const char *fbank_override = getenv("RS_XASR_LOAD_FBANK");
  if (fbank_override && fbank_override[0]) {
    FILE *f = fopen(fbank_override, "rb");
    if (!f) {
      RS_LOG_ERR("XASR: cannot open RS_XASR_LOAD_FBANK=%s", fbank_override);
      return false;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    fbank.resize(sz / sizeof(float));
    size_t rd = fread(fbank.data(), 1, sz, f);
    fclose(f);
    (void)rd;
    n_frames = (int)(fbank.size() / hp_.feature_dim);
    RS_LOG_INFO("XASR: loaded fbank override [%d, %d]", n_frames,
                hp_.feature_dim);
  } else {
    // input_frames = raw PCM in [-1, 1] (use_external_frontend)
    audio_proc_->Compute(input_frames, fbank);
    n_frames = (int)(fbank.size() / hp_.feature_dim);
  }
  if (n_frames <= 0) return false;

  return EncodeOffline(fbank, n_frames, st, sched);
}

bool XASRModel::Decode(RSState &state, ggml_backend_sched_t sched) {
  auto &st = static_cast<XASRState &>(state);
  st.tokens.clear();
  st.ctx_ids.assign(hp_.context_size, hp_.blank_id);
  RunPredictor(st.ctx_ids.data(), st.dec_out);
  return GreedyDecodeRange(st, 0, sched);
}

// greedy transducer over enc_proj rows [from_frame, n_enc_frames),
// continuing st.ctx_ids / st.dec_out / st.tokens. Runs on the model's own
// dec_sched_ so it never disturbs the persistent streaming encoder graph.
bool XASRModel::GreedyDecodeRange(XASRState &st, int from_frame,
                                  ggml_backend_sched_t sched) {
  if (dec_sched_) sched = dec_sched_;
  const int T = st.n_enc_frames;
  const int J = hp_.joiner_dim;
  const int V = hp_.vocab_size;
  if (T <= from_frame) return true;

  int t = from_frame;
  while (t < T) {
    const int seg = T - t;
    ggml_backend_sched_reset(sched);
    if (ctx_) {
      ggml_free(ctx_);
      ctx_ = nullptr;
      gf_ = nullptr;
    }
    if (!init_compute_ctx(&ctx_, &gf_, 64)) return false;

    struct ggml_tensor *enc_seg =
        ggml_new_tensor_2d(ctx_, GGML_TYPE_F32, J, seg);
    struct ggml_tensor *dec_in = ggml_new_tensor_1d(ctx_, GGML_TYPE_F32, J);
    ggml_set_input(enc_seg);
    ggml_set_input(dec_in);

    struct ggml_tensor *dec_proj = ggml_add(
        ctx_, ggml_mul_mat(ctx_, joiner_dec_proj_w, dec_in), joiner_dec_proj_b);
    struct ggml_tensor *act =
        ggml_tanh(ctx_, ggml_add(ctx_, enc_seg, dec_proj));
    struct ggml_tensor *lg = ggml_add(
        ctx_, ggml_mul_mat(ctx_, joiner_out_w, act), joiner_out_b); // [V, seg]
    ggml_set_output(lg);
    ggml_build_forward_expand(gf_, lg);

    if (!ggml_backend_sched_alloc_graph(sched, gf_)) return false;
    ggml_backend_tensor_set(enc_seg, st.enc_proj.data() + (size_t)t * J, 0,
                            (size_t)seg * J * sizeof(float));
    ggml_backend_tensor_set(dec_in, st.dec_out.data(), 0, J * sizeof(float));
    if (ggml_backend_sched_graph_compute(sched, gf_) != GGML_STATUS_SUCCESS) {
      return false;
    }

    // scan for first frame whose argmax != blank
    int emit_frame = -1, emit_tok = -1;
    std::vector<float> seg_logits((size_t)seg * V);
    ggml_backend_tensor_get(lg, seg_logits.data(), 0,
                            seg_logits.size() * sizeof(float));
    for (int i = 0; i < seg && emit_frame < 0; i++) {
      const float *row = seg_logits.data() + (size_t)i * V;
      int best = 0;
      float bv = row[0];
      for (int v = 1; v < V; v++) {
        if (row[v] > bv) {
          bv = row[v];
          best = v;
        }
      }
      if (best != hp_.blank_id) {
        emit_frame = i;
        emit_tok = best;
      }
    }

    ggml_free(ctx_);
    ctx_ = nullptr;
    gf_ = nullptr;

    if (emit_frame < 0) break; // all blank to the end
    st.tokens.push_back(emit_tok);
    for (int j = 0; j + 1 < hp_.context_size; j++)
      st.ctx_ids[j] = st.ctx_ids[j + 1];
    st.ctx_ids[hp_.context_size - 1] = emit_tok;
    RunPredictor(st.ctx_ids.data(), st.dec_out);
    t += emit_frame + 1;
  }
  return true;
}

// ---------------------------------------------------------------------------
// streaming driver: PCM in -> fbank windows -> chunk graphs -> greedy
// ---------------------------------------------------------------------------

bool XASRModel::EncodeStreamingChunk(const float *pcm, size_t n_samples,
                                     XASRState &st,
                                     ggml_backend_sched_t sched) {
  if (!st.stream_initialized) InitStreamState(st);
  if (pcm && n_samples) {
    st.pcm_buffer.insert(st.pcm_buffer.end(), pcm, pcm + n_samples);
  }

  const int T_win = chunk_len_ + hp_.pad_length; // e.g. 32 + 13
  const int ready = audio_proc_->NumReadyFrames(st.pcm_buffer.size());

  using clk = std::chrono::steady_clock;
  auto secs = [](clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double>(b - a).count();
  };
  while (st.fbank_frames_done + T_win <= ready) {
    const int decode_from = st.n_enc_frames;
    std::vector<float> fbank_chunk;
    auto t0 = clk::now();
    audio_proc_->ComputeFbankFrames(st.pcm_buffer, st.fbank_frames_done, T_win,
                                    fbank_chunk);
    auto t1 = clk::now();
    if (!EncodeStreamChunkGraph(fbank_chunk, st, sched)) return false;
    st.fbank_frames_done += chunk_len_;
    auto t2 = clk::now();
    if (!GreedyDecodeRange(st, decode_from, sched)) return false;
    auto t3 = clk::now();
    st.t_fbank += secs(t0, t1);
    st.t_encode += secs(t1, t2);
    st.t_decode += secs(t2, t3);
    st.n_chunks++;
  }
  return true;
}

bool XASRModel::FinishStream(XASRState &st, ggml_backend_sched_t sched) {
  if (!st.stream_initialized) return true;
  // pad ~1 s of silence so trailing speech gets enough right context to be
  // emitted (official sherpa clients add ~0.66 s tail padding similarly)
  const size_t tail =
      (size_t)16000 + (size_t)(chunk_len_ + hp_.pad_length) * 160;
  std::vector<float> zeros(tail, 0.0f);
  const bool ok = EncodeStreamingChunk(zeros.data(), zeros.size(), st, sched);
  if (st.n_chunks > 0) {
    RS_LOG_INFO("XASR stream: %d chunks, per-chunk fbank %.2fms encode %.2fms "
                "decode %.2fms",
                st.n_chunks, 1e3 * st.t_fbank / st.n_chunks,
                1e3 * st.t_encode / st.n_chunks,
                1e3 * st.t_decode / st.n_chunks);
  }
  return ok;
}

// ---------------------------------------------------------------------------
// detokenization (BPE ▁ -> space, CJK spacing cleanup like the official
// sherpa_streaming_infer.py _normalize_cjk_spacing)
// ---------------------------------------------------------------------------

static bool is_cjk_cp(uint32_t cp) {
  return (cp >= 0x3400 && cp <= 0x4dbf) || (cp >= 0x4e00 && cp <= 0x9fff) ||
         (cp >= 0xf900 && cp <= 0xfaff) || (cp >= 0x3000 && cp <= 0x303f) ||
         (cp >= 0xff00 && cp <= 0xffef);
}

static std::vector<uint32_t> utf8_to_cps(const std::string &s) {
  std::vector<uint32_t> cps;
  for (size_t i = 0; i < s.size();) {
    const unsigned char c = s[i];
    uint32_t cp = 0;
    int n = 1;
    if (c < 0x80) {
      cp = c;
    } else if ((c >> 5) == 0x6) {
      cp = c & 0x1f;
      n = 2;
    } else if ((c >> 4) == 0xe) {
      cp = c & 0x0f;
      n = 3;
    } else {
      cp = c & 0x07;
      n = 4;
    }
    for (int k = 1; k < n && i + k < s.size(); k++) {
      cp = (cp << 6) | (s[i + k] & 0x3f);
    }
    cps.push_back(cp);
    i += n;
  }
  return cps;
}

static void cp_to_utf8(uint32_t cp, std::string &out) {
  if (cp < 0x80) {
    out += (char)cp;
  } else if (cp < 0x800) {
    out += (char)(0xc0 | (cp >> 6));
    out += (char)(0x80 | (cp & 0x3f));
  } else if (cp < 0x10000) {
    out += (char)(0xe0 | (cp >> 12));
    out += (char)(0x80 | ((cp >> 6) & 0x3f));
    out += (char)(0x80 | (cp & 0x3f));
  } else {
    out += (char)(0xf0 | (cp >> 18));
    out += (char)(0x80 | ((cp >> 12) & 0x3f));
    out += (char)(0x80 | ((cp >> 6) & 0x3f));
    out += (char)(0x80 | (cp & 0x3f));
  }
}

std::string XASRModel::GetTranscription(RSState &state) {
  auto &st = static_cast<XASRState &>(state);
  std::string raw;
  for (int id : st.tokens) {
    auto it = id_to_token_.find(id);
    if (it == id_to_token_.end()) continue;
    raw += it->second;
  }
  // ▁ (U+2581, e2 96 81) -> space
  std::string text;
  for (size_t i = 0; i < raw.size();) {
    if (i + 2 < raw.size() && (unsigned char)raw[i] == 0xe2 &&
        (unsigned char)raw[i + 1] == 0x96 && (unsigned char)raw[i + 2] == 0x81) {
      text += ' ';
      i += 3;
    } else {
      text += raw[i];
      i += 1;
    }
  }
  // CJK spacing cleanup: drop spaces between CJK chars, around CJK
  // punctuation, and before ASCII punctuation (mirrors the official
  // _normalize_cjk_spacing post-processing).
  std::vector<uint32_t> cps = utf8_to_cps(text);
  std::vector<uint32_t> ocps;
  const std::string ascii_punct = ",.!?;:%)]}";
  for (size_t i = 0; i < cps.size(); i++) {
    if (cps[i] != ' ') {
      ocps.push_back(cps[i]);
      continue;
    }
    uint32_t prev = ocps.empty() ? 0 : ocps.back();
    uint32_t next = 0;
    for (size_t j = i + 1; j < cps.size(); j++) {
      if (cps[j] != ' ') {
        next = cps[j];
        break;
      }
    }
    const bool next_ascii_punct =
        next < 128 && ascii_punct.find((char)next) != std::string::npos;
    const bool drop = prev == 0 || next == 0 ||
                      (is_cjk_cp(prev) && is_cjk_cp(next)) || next_ascii_punct;
    if (!drop && (ocps.empty() || ocps.back() != ' ')) ocps.push_back(' ');
  }
  std::string out;
  for (uint32_t cp : ocps) cp_to_utf8(cp, out);
  // trim trailing spaces
  while (!out.empty() && out.back() == ' ') out.pop_back();
  return out;
}

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------

void rs_register_xasr() {
  rs_register_model_arch("x-asr",
                         []() { return std::make_shared<XASRModel>(); });
  rs_register_model_arch("xasr",
                         []() { return std::make_shared<XASRModel>(); });
}

static struct XASRRegistrar {
  XASRRegistrar() { rs_register_xasr(); }
} _xasr_registrar;
