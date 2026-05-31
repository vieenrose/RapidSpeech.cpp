#pragma once

#include "llm_graph.h"
#include "llm_model.h"
#include <memory>

/**
 * Qwen2 graph builder.
 *
 * Differs from Qwen3:
 *   - Q/K/V projections carry a bias term (added after the matmul).
 *   - No Q/K RMS-norm.
 *
 * Everything else (RMSNorm, RoPE-NEOX, SwiGLU FFN, GQA attention, KV cache,
 * causal mask, lm_head slicing for prefill) is identical to Qwen3, so the
 * implementation reuses the same modular helpers from `llm_graph_builder`.
 */
class llm_build_qwen2 : public llm_graph_builder {
public:
  llm_build_qwen2(const llm_model &model, const llm_cparams &cparams,
                  ggml_backend_sched_t sched);

  ~llm_build_qwen2() override = default;

  llm_graph_result_ptr
  build_graph(const int32_t *tokens, uint32_t n_tokens,
              llm_kv_cache *kv_cache = nullptr, const llm_pos *pos = nullptr,
              const llm_build_opts *opts = nullptr) override;

  llm_graph_result_ptr
  build_graph_from_embeds(ggml_tensor *embeds, uint32_t n_tokens,
                          llm_kv_cache *kv_cache = nullptr,
                          const llm_pos *pos = nullptr,
                          const llm_build_opts *opts = nullptr) override;

private:
  const llm_model &model_;

  ggml_tensor *build_embedding_layer(ggml_context *ctx, const int32_t *tokens,
                                     uint32_t n_tokens);
  ggml_tensor *build_transformer_layer(ggml_context *ctx, ggml_tensor *cur,
                                       int32_t il, llm_kv_cache *kv_cache,
                                       ggml_tensor *positions,
                                       ggml_tensor *causal_mask,
                                       uint32_t n_tokens);
  ggml_tensor *build_attention_layer(ggml_context *ctx, ggml_tensor *cur,
                                     int32_t il, llm_kv_cache *kv_cache,
                                     ggml_tensor *positions,
                                     ggml_tensor *causal_mask,
                                     uint32_t n_tokens);
  ggml_tensor *build_ffn_layer(ggml_context *ctx, ggml_tensor *cur, int32_t il);

  ggml_tensor *build_position_tensor(ggml_context *ctx, const llm_pos *pos,
                                     uint32_t n_tokens);
  ggml_tensor *build_causal_mask_tensor(ggml_context *ctx, uint32_t n_tokens,
                                        uint32_t n_kv_cache = 0);

  bool should_extract_layer(int32_t il) const;
};

/**
 * Register Qwen2 graph builder under LLM_ARCH_QWEN2 (only effective when
 * a model has `hparams.use_qkv_bias = true`; the dispatcher inside
 * llm_graph.cpp routes there).
 */
void llm_register_qwen2();
