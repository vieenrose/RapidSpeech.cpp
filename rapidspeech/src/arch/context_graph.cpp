#include "context_graph.h"

#include <algorithm>
#include <queue>

namespace rs {

void ContextGraph::Build(const std::vector<std::vector<int32_t>> &token_ids,
                         const std::vector<float> &scores,
                         const std::vector<std::string> &phrases,
                         const std::vector<float> &ac_thresholds) {
  for (int32_t i = 0; i < static_cast<int32_t>(token_ids.size()); ++i) {
    auto node = root_.get();
    float score = scores.empty() ? 0.0f : scores[i];
    score = score == 0.0f ? context_score_ : score;
    float ac_threshold = ac_thresholds.empty() ? 0.0f : ac_thresholds[i];
    ac_threshold = ac_threshold == 0.0f ? ac_threshold_ : ac_threshold;
    std::string phrase = phrases.empty() ? std::string() : phrases[i];

    const auto &ids = token_ids[i];
    for (int32_t j = 0; j < static_cast<int32_t>(ids.size()); ++j) {
      int32_t token = ids[j];
      bool is_end = (j == static_cast<int32_t>(ids.size()) - 1);
      auto it = node->next.find(token);
      if (it == node->next.end()) {
        node->next[token] = std::make_unique<ContextState>(
            token, score, node->node_score + score,
            is_end ? node->node_score + score : 0.0f, j + 1,
            is_end ? ac_threshold : 0.0f, is_end,
            is_end ? phrase : std::string());
      } else {
        ContextState *child = it->second.get();
        float new_token_score = std::max(score, child->token_score);
        child->token_score = new_token_score;
        child->node_score = node->node_score + new_token_score;
        bool now_end = is_end || child->is_end;
        child->output_score = now_end ? child->node_score : 0.0f;
        child->is_end = now_end;
        if (is_end) {
          child->phrase = phrase;
          child->ac_threshold = ac_threshold;
        }
      }
      node = node->next[token].get();
    }
  }
  FillFailOutput();
}

std::tuple<float, const ContextState *, const ContextState *>
ContextGraph::ForwardOneStep(const ContextState *state, int32_t token,
                             bool strict_mode) const {
  const ContextState *node = nullptr;
  float score = 0.0f;
  auto it = state->next.find(token);
  if (it != state->next.end()) {
    node = it->second.get();
    score = node->token_score;
  } else {
    node = state->fail;
    while (node->next.find(token) == node->next.end()) {
      node = node->fail;
      if (node->token == -1) break; // root
    }
    auto it2 = node->next.find(token);
    if (it2 != node->next.end()) {
      node = it2->second.get();
    }
    score = node->node_score - state->node_score;
  }

  const ContextState *matched =
      node->is_end ? node : (node->output != nullptr ? node->output : nullptr);

  if (!strict_mode && node->output_score != 0.0f) {
    float output_score = node->is_end
                             ? node->node_score
                             : (node->output != nullptr
                                    ? node->output->node_score
                                    : node->node_score);
    return std::make_tuple(score + output_score - node->node_score,
                           root_.get(), matched);
  }
  return std::make_tuple(score + node->output_score, node, matched);
}

std::pair<bool, const ContextState *>
ContextGraph::IsMatched(const ContextState *state) const {
  if (state->is_end) return {true, state};
  if (state->output != nullptr) return {true, state->output};
  return {false, nullptr};
}

std::pair<float, const ContextState *>
ContextGraph::Finalize(const ContextState *state) const {
  return {-state->node_score, root_.get()};
}

void ContextGraph::FillFailOutput() {
  std::queue<ContextState *> q;
  for (auto &kv : root_->next) {
    kv.second->fail = root_.get();
    q.push(kv.second.get());
  }
  while (!q.empty()) {
    auto cur = q.front();
    q.pop();
    for (auto &kv : cur->next) {
      int32_t tok = kv.first;
      const ContextState *fail = cur->fail;
      if (fail->next.find(tok) != fail->next.end()) {
        fail = fail->next.at(tok).get();
      } else {
        fail = fail->fail;
        while (fail->next.find(tok) == fail->next.end()) {
          fail = fail->fail;
          if (fail->token == -1) break;
        }
        auto it = fail->next.find(tok);
        if (it != fail->next.end()) fail = it->second.get();
      }
      kv.second->fail = fail;

      const ContextState *output = fail;
      while (!output->is_end) {
        output = output->fail;
        if (output->token == -1) {
          output = nullptr;
          break;
        }
      }
      kv.second->output = output;
      kv.second->output_score += output == nullptr ? 0.0f : output->output_score;
      q.push(kv.second.get());
    }
  }
}

} // namespace rs
