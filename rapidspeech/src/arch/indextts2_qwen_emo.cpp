// IndexTTS-2 QwenEmotion: Qwen3-0.6B emotion classifier.
//
// Mirrors infer_v2.py QwenEmotion. Given an emotion description text, runs the
// Qwen3 chat model with system prompt "文本情感分类", parses the JSON output and
// returns an 8-d emotion vector in the order:
//   [happy(高兴), angry(愤怒), sad(悲伤), afraid(恐惧),
//    disgusted(反感), melancholic(低落), surprised(惊讶), calm(自然)]
//
// The model GGUF is shipped separately (convert_hf_to_gguf.py) and loaded lazily
// via the env var RS_INDEXTTS2_QWEN_EMO_GGUF. Generation uses a full-sequence
// greedy recompute (no KV cache) — the output is a short JSON, so this is cheap
// and avoids the GPU-KV decode machinery used by qwen3_asr.cpp.

#include "indextts2.h"

#include "arch/llm_model.h"
#include "arch/llm_graph.h"
#include "arch/qwen3.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <regex>
#include <string>
#include <map>
#include <vector>

namespace indextts2 {

namespace {

// Emotion key order — MUST match infer_v2.py QwenEmotion.desired_vector_order.
static const char *kCnKeys[8] = {
    "\xE9\xAB\x98\xE5\x85\xB4",         // 高兴 happy
    "\xE6\x84\xA4\xE6\x80\x92",         // 愤怒 angry
    "\xE6\x82\xB2\xE4\xBC\xA4",         // 悲伤 sad
    "\xE6\x81\x90\xE6\x83\xA7",         // 恐惧 afraid
    "\xE5\x8F\x8D\xE6\x84\x9F",         // 反感 disgusted
    "\xE4\xBD\x8E\xE8\x90\xBD",         // 低落 melancholic
    "\xE6\x83\x8A\xE8\xAE\xB6",         // 惊讶 surprised
    "\xE8\x87\xAA\xE7\x84\xB6",         // 自然 calm
};

// Words that force the "悲伤"(sad) detection to mean "低落"(melancholic) instead.
static const char *kMelancholicWords[] = {
    "\xE4\xBD\x8E\xE8\x90\xBD",  // 低落
    "melancholy", "melancholic", "depression", "depressed", "gloomy",
};

constexpr float kMaxScore = 1.2f;
constexpr float kMinScore = 0.0f;

static float clamp_score(float v) {
    return std::max(kMinScore, std::min(kMaxScore, v));
}

// Find `key` in `content` and parse the float that follows the next ':'.
// Returns true and writes `out` if found.
static bool parse_key_value(const std::string &content, const std::string &key,
                            float &out) {
    size_t pos = content.find(key);
    if (pos == std::string::npos) return false;
    size_t colon = content.find(':', pos + key.size());
    if (colon == std::string::npos) return false;
    size_t i = colon + 1;
    while (i < content.size() &&
           (content[i] == ' ' || content[i] == '"' || content[i] == '\t'))
        ++i;
    size_t j = i;
    while (j < content.size() &&
           (std::isdigit((unsigned char)content[j]) || content[j] == '.' ||
            content[j] == '-' || content[j] == 'e' || content[j] == 'E' ||
            content[j] == '+'))
        ++j;
    if (j == i) return false;
    try {
        out = std::stof(content.substr(i, j - i));
    } catch (...) {
        return false;
    }
    return true;
}

} // namespace

bool Model::LoadQwenEmotion(const std::string &path, ggml_backend_t backend) {
    if (qwen_emo_ready_) return true;
    if (path.empty()) return false;
    qwen_emo_model_ = std::make_shared<llm_model>();
    if (!qwen_emo_model_->load_from_file(path, backend)) {
        RS_LOG_ERR("[indextts2] LoadQwenEmotion: load_from_file failed: %s",
                   path.c_str());
        qwen_emo_model_.reset();
        return false;
    }
    if (qwen_emo_model_->arch() != LLM_ARCH_QWEN3) {
        RS_LOG_ERR("[indextts2] LoadQwenEmotion: expected Qwen3 arch (got %d)",
                   (int)qwen_emo_model_->arch());
        qwen_emo_model_.reset();
        return false;
    }
    // load_from_file only loads tensors into the weight context; it does NOT
    // wire them into tok_embd_/layers_. Build the name→tensor map and run the
    // Qwen3 mapping (mirrors qwen3_asr.cpp), otherwise build_graph's get_rows
    // dereferences a null tok_embd_ and segfaults.
    {
        std::map<std::string, ggml_tensor *> tensors;
        ggml_context *wctx = qwen_emo_model_->get_weight_ctx();
        for (ggml_tensor *t = wctx ? ggml_get_first_tensor(wctx) : nullptr;
             t != nullptr; t = ggml_get_next_tensor(wctx, t)) {
            const char *nm = ggml_get_name(t);
            if (nm && nm[0]) tensors[nm] = t;
        }
        if (!qwen_emo_model_->map_tensors_qwen3(tensors)) {
            RS_LOG_ERR("[indextts2] LoadQwenEmotion: map_tensors_qwen3 failed");
            qwen_emo_model_.reset();
            return false;
        }
    }
    auto &vocab = const_cast<llm_vocab &>(qwen_emo_model_->vocab());
    vocab.load_qwen3_special_tokens();
    qwen_emo_ready_ = true;
    RS_LOG_INFO("[indextts2] QwenEmotion loaded: layers=%d embd=%d vocab=%d",
                qwen_emo_model_->hparams().n_layer,
                qwen_emo_model_->hparams().n_embd,
                qwen_emo_model_->hparams().n_vocab);
    return true;
}

bool Model::QwenEmoInfer(ggml_backend_sched_t sched, const std::string &text,
                         std::vector<float> &vec8) {
    vec8.assign(8, 0.0f);

    // Lazy-load on first use from the env var.
    if (!qwen_emo_ready_) {
        if (const char *p = std::getenv("RS_INDEXTTS2_QWEN_EMO_GGUF")) {
            LoadQwenEmotion(p, backend_);
        }
    }
    if (!qwen_emo_ready_ || !qwen_emo_model_) return false;

    auto &vocab = const_cast<llm_vocab &>(qwen_emo_model_->vocab());

    // qwen0.6bemo4-merge uses a llamafactory chat template (NOT ChatML):
    //   System: <sys><|endoftext|>\nHuman: <text><|endoftext|>\nAssistant:
    // System prompt is "文本情感分类". <|endoftext|> (151643) is the separator
    // and the stop token.
    std::string prompt =
        "System: \xE6\x96\x87\xE6\x9C\xAC\xE6\x83\x85\xE6\x84\x9F\xE5\x88\x86"
        "\xE7\xB1\xBB<|endoftext|>\n"                  // System: 文本情感分类
        "Human: " + text + "<|endoftext|>\nAssistant:";

    std::vector<int32_t> tokens = vocab.tokenize(prompt, /*add_bos=*/false);
    if (tokens.empty()) {
        RS_LOG_ERR("[indextts2] QwenEmoInfer: prompt tokenization empty");
        return false;
    }

    int32_t eot = vocab.find_token_id("<|endoftext|>");
    const int32_t eos = vocab.token_eos();
    if (eot < 0) eot = eos;

    llm_cparams cparams;
    cparams.n_ctx          = 4096;
    cparams.n_batch        = 4096;
    cparams.n_ubatch       = 4096;
    cparams.n_threads      = 4;
    cparams.n_threads_batch = 4;
    auto builder = std::make_unique<llm_build_qwen3>(*qwen_emo_model_, cparams,
                                                     sched);

    constexpr int kMaxNew = 256;
    std::vector<int32_t> generated;
    for (int step = 0; step < kMaxNew; ++step) {
        llm_build_opts opts;
        opts.output_mode    = llm_output_mode::OUTPUT_LOGITS;
        opts.skip_embeddings = false;
        opts.use_kv_cache    = false;
        opts.causal_mask     = true;
        opts.is_decode_step  = false;

        ggml_backend_sched_reset(sched);
        auto result = builder->build_graph((const int32_t *)tokens.data(),
                                           (uint32_t)tokens.size(), nullptr,
                                           nullptr, &opts);
        if (!result) {
            RS_LOG_ERR("[indextts2] QwenEmoInfer: build_graph failed");
            return false;
        }
        if (!ggml_backend_sched_alloc_graph(sched, result->get_graph())) {
            RS_LOG_ERR("[indextts2] QwenEmoInfer: alloc graph failed");
            return false;
        }
        if (auto t = result->get_input_tensor("inp_tokens")) {
            ggml_backend_tensor_set(t, tokens.data(), 0,
                                    tokens.size() * sizeof(int32_t));
        }
        if (auto t = result->get_input_tensor("position_ids")) {
            std::vector<llm_pos> pos(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) pos[i] = (llm_pos)i;
            result->set_position_ids(t, pos.data(), (uint32_t)tokens.size());
        } else if (auto t2 = result->get_input_tensor("position_ids_seq")) {
            std::vector<llm_pos> pos(tokens.size());
            for (size_t i = 0; i < tokens.size(); ++i) pos[i] = (llm_pos)i;
            result->set_position_ids(t2, pos.data(), (uint32_t)tokens.size());
        }
        if (auto t = result->get_input_tensor("causal_mask")) {
            result->set_causal_mask(t, (uint32_t)tokens.size(), 0);
        }
        if (ggml_backend_sched_graph_compute(sched, result->get_graph()) !=
            GGML_STATUS_SUCCESS) {
            RS_LOG_ERR("[indextts2] QwenEmoInfer: compute failed");
            return false;
        }
        ggml_tensor *logits = result->get_logits();
        if (!logits) return false;
        const int n_vocab = (int)logits->ne[0];
        // OUTPUT_LOGITS produces logits for ALL positions: [n_vocab, n_tokens].
        // We need the LAST token's row to predict the next token.
        const int n_pos = (int)logits->ne[1];
        const size_t last_off = (size_t)(n_pos - 1) * n_vocab;
        std::vector<float> lh(n_vocab);
        ggml_backend_tensor_get(logits, lh.data(), last_off * sizeof(float),
                                (size_t)n_vocab * sizeof(float));
        int32_t next = 0;
        float best = lh[0];
        for (int v = 1; v < n_vocab; ++v)
            if (lh[v] > best) { best = lh[v]; next = v; }

        if (next == eot || next == eos) break;
        generated.push_back(next);
        tokens.push_back(next);
    }

    std::string content = vocab.detokenize(generated);
    RS_LOG_INFO("[indextts2] QwenEmotion raw output: %s", content.c_str());

    // Parse the 8 emotion scores (JSON or loose "key: value").
    float scores[8] = {0};
    bool any = false;
    for (int i = 0; i < 8; ++i) {
        float v = 0.0f;
        if (parse_key_value(content, kCnKeys[i], v)) {
            scores[i] = clamp_score(v);
            if (scores[i] > 0.0f) any = true;
        }
    }

    // melancholic workaround: swap 悲伤(sad,idx2) ↔ 低落(melancholic,idx5).
    std::string lower = text;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    bool melancholic = false;
    for (const char *w : kMelancholicWords) {
        if (lower.find(w) != std::string::npos) { melancholic = true; break; }
    }
    if (melancholic) std::swap(scores[2], scores[5]);

    // Default to calm/neutral if nothing was detected.
    if (!any) scores[7] = 1.0f;

    for (int i = 0; i < 8; ++i) vec8[i] = scores[i];
    return true;
}

} // namespace indextts2
