// IndexTTS-2 SentencePiece-unigram tokenizer (Viterbi).
//
// Ported from the IndexTTS 1.5 C++ reference (CrispASR/src/indextts.cpp). The
// tokenizer follows the upstream Python pipeline:
//   indextts/utils/front.py:TextNormalizer.char_rep_map  → CJK punct → ASCII
//   indextts/utils/common.py:tokenize_by_CJK_char        → space-separate CJK
//   .upper()                                             → uppercase ASCII
//   sentencepiece Encode                                 → Viterbi over unigram
//
// Vocab + scores live in GGUF (`tokenizer.ggml.tokens` / `tokenizer.ggml.scores`)
// so we don't link libsentencepiece. The convert script extracts them via
// sentencepiece-python.

#include "indextts2.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace indextts2 {

// ---------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------
static int utf8_decode(const char *s, uint32_t &cp) {
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) { cp = c; return 1; }
    if ((c & 0xE0) == 0xC0) {
        cp = ((uint32_t)(c & 0x1F) << 6) |
             (uint32_t)((unsigned char)s[1] & 0x3F);
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        cp = ((uint32_t)(c & 0x0F) << 12) |
             ((uint32_t)((unsigned char)s[1] & 0x3F) << 6) |
             (uint32_t)((unsigned char)s[2] & 0x3F);
        return 3;
    }
    if ((c & 0xF8) == 0xF0) {
        cp = ((uint32_t)(c & 0x07) << 18) |
             ((uint32_t)((unsigned char)s[1] & 0x3F) << 12) |
             ((uint32_t)((unsigned char)s[2] & 0x3F) << 6) |
             (uint32_t)((unsigned char)s[3] & 0x3F);
        return 4;
    }
    cp = '?';
    return 1;
}

static void utf8_encode(std::string &out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char)cp;
    } else if (cp < 0x800) {
        out += (char)(0xC0 | (cp >> 6));
        out += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char)(0xE0 | (cp >> 12));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    } else {
        out += (char)(0xF0 | (cp >> 18));
        out += (char)(0x80 | ((cp >> 12) & 0x3F));
        out += (char)(0x80 | ((cp >> 6) & 0x3F));
        out += (char)(0x80 | (cp & 0x3F));
    }
}

// CJK_RANGE_PATTERN from upstream indextts/utils/common.py.
static bool is_cjk_codepoint(uint32_t cp) {
    return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
           (cp >= 0xA840 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
           (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
           (cp >= 0x20000 && cp <= 0x2FFFF);
}

// TextNormalizer.char_rep_map — CJK punctuation → ASCII. 0 = no mapping.
static uint32_t normalize_cjk_punct(uint32_t cp) {
    switch (cp) {
    case 0xFF1A: return ',';   // ：
    case 0xFF1B: return ',';   // ；
    case 0xFF0C: return ',';   // ，
    case 0x3002: return '.';   // 。
    case 0xFF01: return '!';   // ！
    case 0xFF1F: return '?';   // ？
    case 0x3001: return ',';   // 、
    case 0x00B7: return '-';   // ·
    case 0x201C: return '\'';  // “
    case 0x201D: return '\'';  // ”
    case 0x2018: return '\'';  // ‘
    case 0x2019: return '\'';  // ’
    case 0x300C: return '\'';  // 「
    case 0x300D: return '\'';  // 」
    case 0xFF08: return '\'';  // （
    case 0xFF09: return '\'';  // ）
    case 0x300A: return '\'';  // 《
    case 0x300B: return '\'';  // 》
    case 0x3010: return '\'';  // 【
    case 0x3011: return '\'';  // 】
    case 0x2014: return '-';   // —
    case 0xFF5E: return '-';   // ～
    default:     return 0;
    }
}

// punct map → tokenize_by_CJK_char (space-separate every CJK char) → upper().
static std::string preprocess_text(const std::string &text) {
    std::string out;
    out.reserve(text.size() * 2);
    bool last_was_sep = true;

    for (size_t i = 0; i < text.size();) {
        uint32_t cp = 0;
        int n = utf8_decode(text.c_str() + i, cp);
        if (n <= 0) break;
        i += (size_t)n;

        if (cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r') {
            if (!last_was_sep) { out += ' '; last_was_sep = true; }
            continue;
        }
        if (uint32_t r = normalize_cjk_punct(cp)) cp = r;

        if (is_cjk_codepoint(cp)) {
            if (!last_was_sep) out += ' ';
            utf8_encode(out, cp);
            out += ' ';
            last_was_sep = true;
        } else {
            if (cp >= 'a' && cp <= 'z') cp -= 32;
            utf8_encode(out, cp);
            last_was_sep = false;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

// ---------------------------------------------------------------------------
// BPETokenizer
// ---------------------------------------------------------------------------
bool BPETokenizer::load_from_vocab(std::vector<std::string> tokens,
                                    std::vector<float> scores_in) {
    id_to_token = std::move(tokens);
    scores      = std::move(scores_in);
    token_to_id.clear();
    token_to_id.reserve(id_to_token.size());
    for (size_t i = 0; i < id_to_token.size(); ++i)
        token_to_id[id_to_token[i]] = (int32_t)i;
    loaded = !id_to_token.empty();
    return loaded;
}

std::vector<int32_t> BPETokenizer::encode(const std::string &text) const {
    const std::string pre = preprocess_text(text);
    if (!loaded || id_to_token.empty()) {
        // Byte-level fallback so callers see a non-empty token stream during
        // bring-up. Clamped to 12000-vocab semantics.
        std::vector<int32_t> r;
        r.reserve(pre.size());
        for (unsigned char c : pre) {
            int32_t id = (int32_t)c;
            if (id >= 12000) id = 0;
            r.push_back(id);
        }
        return r;
    }

    // SentencePiece convention: leading ▁ (U+2581 = 0xE2 0x96 0x81) and
    // ' ' → ▁ inside.
    std::string proc;
    proc += "\xe2\x96\x81";
    for (char c : pre)
        proc += (c == ' ') ? std::string("\xe2\x96\x81") : std::string(1, c);

    const int N = (int)proc.size();
    const int max_piece_len = 48;

    std::vector<float> best_score(N + 1, -1e30f);
    std::vector<int>   best_len(N + 1, 0);
    best_score[0] = 0.0f;

    for (int i = 0; i < N; ++i) {
        if (best_score[i] <= -1e29f) continue;
        for (int len = 1; len <= std::min(max_piece_len, N - i); ++len) {
            std::string piece = proc.substr(i, len);
            auto it = token_to_id.find(piece);
            if (it != token_to_id.end()) {
                float sc = (it->second < (int32_t)scores.size())
                               ? scores[it->second]
                               : -20.0f;
                float tot = best_score[i] + sc;
                if (tot > best_score[i + len]) {
                    best_score[i + len] = tot;
                    best_len[i + len]   = len;
                }
            }
        }
        if (best_score[i + 1] <= -1e29f) {
            best_score[i + 1] = best_score[i] + (-20.0f);
            best_len[i + 1]   = 1;
        }
    }

    std::vector<std::pair<int, int>> segments;
    int pos = N;
    while (pos > 0) {
        int len = best_len[pos];
        segments.push_back({pos - len, len});
        pos -= len;
    }
    std::reverse(segments.begin(), segments.end());

    std::vector<int32_t> result;
    bool prev_unk = false;
    for (const auto &seg : segments) {
        std::string s = proc.substr(seg.first, seg.second);
        auto it = token_to_id.find(s);
        if (it != token_to_id.end()) {
            result.push_back(it->second);
            prev_unk = false;
        } else {
            if (!prev_unk) result.push_back(2); // <unk>
            prev_unk = true;
        }
    }
    return result;
}

} // namespace indextts2
