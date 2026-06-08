#pragma once

// Aho-Corasick trie for keyword/context biasing.
// Ported from sherpa-onnx (sherpa-onnx/csrc/context-graph.h, Apache-2.0).
// Stripped of sherpa-onnx namespace/logging dependencies so it can be reused
// by the CTC-based KWS decoder.

#include "rapidspeech.h" // RS_API
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rs {

class ContextGraph;
using ContextGraphPtr = std::shared_ptr<ContextGraph>;

struct ContextState {
  int32_t token = -1;
  float token_score = 0.0f;
  float node_score = 0.0f;
  float output_score = 0.0f;
  int32_t level = 0;
  float ac_threshold = 0.0f;
  bool is_end = false;
  std::string phrase;
  std::unordered_map<int32_t, std::unique_ptr<ContextState>> next;
  const ContextState *fail = nullptr;
  const ContextState *output = nullptr;

  ContextState() = default;
  ContextState(int32_t t, float ts, float ns, float os, int32_t lv = 0,
               float ac = 0.0f, bool end = false, const std::string &ph = {})
      : token(t), token_score(ts), node_score(ns), output_score(os), level(lv),
        ac_threshold(ac), is_end(end), phrase(ph) {}
};

class RS_API ContextGraph {
public:
  ContextGraph() = default;

  // token_ids[i] is the token-id sequence of the i-th keyword.
  // scores[i] (optional) overrides the default per-token boost for keyword i.
  // phrases[i] (optional) is the human-readable label reported on a match.
  // ac_thresholds[i] (optional) overrides the default avg-prob threshold.
  ContextGraph(const std::vector<std::vector<int32_t>> &token_ids,
               float context_score, float ac_threshold,
               const std::vector<float> &scores = {},
               const std::vector<std::string> &phrases = {},
               const std::vector<float> &ac_thresholds = {})
      : context_score_(context_score), ac_threshold_(ac_threshold) {
    root_ = std::make_unique<ContextState>(-1, 0, 0, 0);
    root_->fail = root_.get();
    Build(token_ids, scores, phrases, ac_thresholds);
  }

  // Advance one step from `state` on `token`.
  // Returns (boost_score_delta, new_state, matched_node_or_nullptr).
  // Strict mode is what we want for CTC: stay on the matched trie node.
  std::tuple<float, const ContextState *, const ContextState *>
  ForwardOneStep(const ContextState *state, int32_t token_id,
                 bool strict_mode = true) const;

  std::pair<bool, const ContextState *>
  IsMatched(const ContextState *state) const;

  std::pair<float, const ContextState *>
  Finalize(const ContextState *state) const;

  const ContextState *Root() const { return root_.get(); }

private:
  float context_score_ = 0.0f;
  float ac_threshold_ = 0.0f;
  std::unique_ptr<ContextState> root_;

  void Build(const std::vector<std::vector<int32_t>> &token_ids,
             const std::vector<float> &scores,
             const std::vector<std::string> &phrases,
             const std::vector<float> &ac_thresholds);
  void FillFailOutput();
};

} // namespace rs
