#include "frontend/matcha_frontend.h"
#include "utils/rs_log.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#ifdef HAVE_ESPEAK
#include <espeak-ng/speak_lib.h>
#include <cstdlib>
#endif

// Faithful port of sherpa-onnx's matcha-tts-lexicon.cc (+ text-utils + phrase-matcher +
// piper-phonemize ProcessPhonemes). Goal: byte-identical token ids to sherpa for the
// matcha-icefall-zh-en model. Pipeline: normalize punctuation -> SplitUtf8 (+merge ASCII
// letters into words) -> remove spaces after punct -> greedy phrase match against the
// lexicon -> per-word ConvertWordToIds (lexicon | token | CJK-recurse | espeak), inserting
// a blank (space) token before a word whenever the previous word started with a letter.

namespace {

// ---- UTF-8 ----
uint32_t utf8_cp(const std::string& s, size_t i, int& len) {
  unsigned char c = (unsigned char)s[i];
  if (c < 0x80) { len = 1; return c; }
  if ((c & 0xE0) == 0xC0 && i + 1 < s.size()) { len = 2; return ((c & 0x1F) << 6) | ((unsigned char)s[i+1] & 0x3F); }
  if ((c & 0xF0) == 0xE0 && i + 2 < s.size()) { len = 3; return ((c & 0x0F) << 12) | (((unsigned char)s[i+1] & 0x3F) << 6) | ((unsigned char)s[i+2] & 0x3F); }
  if ((c & 0xF8) == 0xF0 && i + 3 < s.size()) { len = 4; return ((c & 0x07) << 18) | (((unsigned char)s[i+1] & 0x3F) << 12) | (((unsigned char)s[i+2] & 0x3F) << 6) | ((unsigned char)s[i+3] & 0x3F); }
  len = 1; return 0xFFFD;
}

// sherpa text-utils IsCJK
bool is_cjk_cp(uint32_t cp) {
  return (cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x2E80 && cp <= 0xA4CF) ||
         (cp >= 0xA840 && cp <= 0xD7AF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
         (cp >= 0xFE30 && cp <= 0xFE4F) || (cp >= 0xFF65 && cp <= 0xFFDC) ||
         (cp >= 0x20000 && cp <= 0x2FFFF);
}
bool contains_cjk(const std::string& s) {
  for (size_t i = 0; i < s.size(); ) { int l; uint32_t cp = utf8_cp(s, i, l); if (is_cjk_cp(cp)) return true; i += l; }
  return false;
}
bool is_punct_byte(unsigned char c) { return c != '\'' && std::ispunct(c); }          // text-utils IsPunct(char)
bool is_alpha_or_punct(unsigned char c) { return std::isalpha(c) || std::ispunct(c); } // IsAlphaOrPunct
bool is_punct_str(const std::string& s) {                                              // IsPunct(string)
  static const std::unordered_set<std::string> p = {
    ",", ".", "!", "?", ":", "\"", "'", "，", "。", "！", "？", "\xe2\x80\x9c", "\xe2\x80\x9d", "\xe2\x80\x98", "\xe2\x80\x99" };
  return p.count(s) > 0;
}
// IsSpecial: european diacritics + french apostrophe (rare for zh/en; minimal port).
bool is_special(const std::string& w) {
  if (w.size() == 3) { auto c0 = (unsigned char)w[0], c1 = (unsigned char)w[1], c2 = (unsigned char)w[2];
    if (c0 == 0xe2 && c1 == 0x80 && c2 == 0x99) return true; }
  if (w.size() == 2) { unsigned char c0 = (unsigned char)w[0]; if (c0 == 0xC3 || c0 == 0xC4 || c0 == 0xC5) return true; }
  return false;
}

// SplitUtf8 = split into codepoints, then merge consecutive ASCII letters into words.
std::vector<std::string> split_chars(const std::string& s) {
  std::vector<std::string> a;
  for (size_t i = 0; i < s.size(); ) { int l; utf8_cp(s, i, l); a.emplace_back(s, i, l); i += l; }
  return a;
}
std::vector<std::string> merge_chars_into_words(const std::vector<std::string>& words) {
  std::vector<std::string> ans; int n = (int)words.size(), i = 0, prev = -1;
  while (i < n) {
    const std::string& w = words[i];
    if (w.size() >= 3 || (w.size() == 2 && !is_special(w)) ||
        (w.size() == 1 && (is_punct_byte((unsigned char)w[0]) || std::isspace((unsigned char)w[0])))) {
      if (prev != -1) { std::string t; for (; prev < i; ++prev) t += words[prev]; prev = -1; ans.push_back(std::move(t)); }
      if (!std::isspace((unsigned char)w[0])) ans.push_back(w);
      ++i; continue;
    }
    if (w.size() == 1 || (w.size() == 2 && is_special(w))) { if (prev == -1) prev = i; ++i; continue; }
    ++i;  // ignore invalid
  }
  if (prev != -1) { std::string t; for (; prev < n; ++prev) t += words[prev]; ans.push_back(std::move(t)); }
  return ans;
}
std::vector<std::string> split_utf8(const std::string& s) { return merge_chars_into_words(split_chars(s)); }

// Normalize: full-width punctuation -> half-width (sherpa replace_str_pairs; note ： and :
// both -> ","), and collapse any whitespace run to a single space.
std::string normalize_text(std::string t) {
  static const std::pair<const char*, const char*> rep[] = {
    {"\xef\xbc\x8c", ","}, {"\xe3\x80\x81", ","}, {"\xef\xbc\x9b", ";"}, {"\xef\xbc\x9a", ","},
    {":", ","}, {"\xe3\x80\x82", "."}, {"\xef\xbc\x9f", "?"}, {"\xef\xbc\x81", "!"} };
  for (const auto& r : rep) { std::string from = r.first, to = r.second; size_t pos = 0;
    while ((pos = t.find(from, pos)) != std::string::npos) { t.replace(pos, from.size(), to); pos += to.size(); } }
  std::string out; bool insp = false;
  for (char c : t) { if (std::isspace((unsigned char)c)) { if (!insp) { out += ' '; insp = true; } } else { out += c; insp = false; } }
  return out;
}

// Remove spaces after punctuation / dedup (sherpa words2 loop).
std::vector<std::string> remove_spaces_after_punct(const std::vector<std::string>& w2) {
  std::vector<std::string> w; w.reserve(w2.size());
  for (size_t i = 0; i < w2.size(); ++i) {
    if (i == 0) w.push_back(w2[i]);
    else if (w2[i] == " ") { if (w.back() == " " || is_punct_str(w.back())) continue; else w.push_back(w2[i]); }
    else if (is_punct_str(w2[i])) { if (w.back() == " " || is_punct_str(w.back())) continue; else w.push_back(w2[i]); }
    else w.push_back(w2[i]);
  }
  return w;
}

// PhraseMatcher: greedy longest phrase (<= max_len words) present in the lexicon.
std::vector<std::string> phrase_match(const std::unordered_set<std::string>& lex,
                                      const std::vector<std::string>& words, int max_len) {
  std::vector<std::string> phrases; int n = (int)words.size(), i = 0;
  while (i < n) {
    int start = i; std::string w;
    if (!words[i].empty() && !is_alpha_or_punct((unsigned char)words[i].front())) {
      int end = std::min(i + max_len - 1, n - 1);
      while (end > start) {
        std::string tw; for (int k = start; k <= end; ++k) tw += words[k];
        if (!tw.empty() && is_alpha_or_punct((unsigned char)tw.back())) { --end; continue; }
        if (lex.count(tw)) { w = tw; i = end + 1; break; }
        --end;
      }
    }
    if (w.empty()) { w = words[i]; ++i; }
    phrases.push_back(std::move(w));
  }
  return phrases;
}

#ifdef HAVE_ESPEAK
// English -> espeak-ng IPA phonemes, run through the model's replacement table (sherpa
// PR #2853 kReplacements), then split into UTF-8 codepoints (the model's tokens).
bool espeak_ready() {
  static int state = -1;
  if (state < 0) {
    const char* dp = std::getenv("ESPEAK_DATA_PATH");
    int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dp, 0);
    state = (sr > 0 && espeak_SetVoiceByName("en-us") == EE_OK) ? 1 : 0;
    if (!state) RS_LOG_ERR("matcha frontend: espeak_Initialize failed (set ESPEAK_DATA_PATH)");
  }
  return state == 1;
}
const std::pair<const char*, const char*> kReplacements[] = {
  {"\xc9\x9d", "\xc9\x9c\xc9\xb9"}, {"\xc9\x9a", "\xc9\x99\xc9\xb9"},   // ɝ->ɜɹ, ɚ->əɹ
  {"e\xc9\xaa", "A"}, {"a\xc9\xaa", "I"}, {"\xc9\x94\xc9\xaa", "Y"},     // eɪ->A, aɪ->I, ɔɪ->Y
  {"o\xca\x8a", "O"}, {"\xc9\x99\xca\x8a", "O"}, {"a\xca\x8a", "W"},      // oʊ->O, əʊ->O, aʊ->W
  {"t\xca\x83", "\xca\xa7"}, {"d\xca\x92", "\xca\xa4"},                  // tʃ->ʧ, dʒ->ʤ
  {"\xcb\x90", ""},                                                       // ː-> (drop)
  {"g", "\xc9\xa1"}, {"r", "\xc9\xb9"},                                  // g->ɡ, r->ɹ
  {"e", "\xc9\x9b"},                                                      // e->ɛ
};
void english_to_codepoints(const std::string& word, std::vector<std::string>& out) {
  if (!espeak_ready()) return;
  std::string ipa;
  const char* text = word.c_str();
  const void* tp = (const void*)text;
  while (tp) { const char* ph = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8, espeakPHONEMES_IPA); if (ph) ipa += ph; if (!ph) break; }
  for (const auto& r : kReplacements) {
    std::string from = r.first, to = r.second; size_t pos = 0;
    while ((pos = ipa.find(from, pos)) != std::string::npos) { ipa.replace(pos, from.size(), to); pos += to.size(); }
  }
  for (size_t i = 0; i < ipa.size(); ) { int l; utf8_cp(ipa, i, l); out.emplace_back(ipa, i, l); i += l; }
}
#endif

}  // namespace

bool MatchaFrontend::Load(const char* tokens_path, const char* lexicon_path) {
  if (!tokens_path || !lexicon_path) { RS_LOG_ERR("matcha frontend: null path"); return false; }
  token2id_.clear(); lexicon_.clear(); all_words_.clear();

  std::ifstream tf(tokens_path);
  if (!tf) { RS_LOG_ERR("matcha frontend: cannot open tokens %s", tokens_path); return false; }
  std::string line;
  while (std::getline(tf, line)) {
    std::istringstream is(line); std::vector<std::string> f; std::string w;
    while (is >> w) f.push_back(w);
    if (f.empty()) continue;
    if (f.size() == 1) token2id_[" "] = std::stoi(f[0]);
    else               token2id_[f[0]] = std::stoi(f[1]);
  }
  if (token2id_.empty()) { RS_LOG_ERR("matcha frontend: empty tokens %s", tokens_path); return false; }

  std::ifstream lf(lexicon_path);
  if (!lf) { RS_LOG_ERR("matcha frontend: cannot open lexicon %s", lexicon_path); return false; }
  while (std::getline(lf, line)) {
    std::istringstream is(line); std::string word; if (!(is >> word)) continue;
    if (lexicon_.count(word)) continue;  // first occurrence wins (sherpa)
    std::vector<int32_t> ids; std::string t; bool ok = true;
    while (is >> t) { auto it = token2id_.find(t); if (it == token2id_.end()) { ok = false; break; } ids.push_back(it->second); }
    if (ok && !ids.empty()) lexicon_[word] = std::move(ids);
  }
  if (lexicon_.empty()) { RS_LOG_ERR("matcha frontend: empty lexicon %s", lexicon_path); return false; }
  for (const auto& kv : lexicon_) all_words_.insert(kv.first);
  RS_LOG_INFO("matcha frontend: %d tokens, %d lexicon entries", (int)token2id_.size(), (int)lexicon_.size());
  return true;
}

std::vector<int32_t> MatchaFrontend::ConvertWordToIds(const std::string& w) const {
  auto lit = lexicon_.find(w);
  if (lit != lexicon_.end()) return lit->second;
  auto tit = token2id_.find(w);
  if (tit != token2id_.end()) return { tit->second };
  std::vector<int32_t> ans;
  if (contains_cjk(w)) {
    // split into per-char words; recurse ONLY into chars present in the lexicon (sherpa) —
    // OOV CJK (e.g. brackets 「」) is silently dropped, and avoids infinite recursion.
    for (auto& ch : split_utf8(w)) {
      if (!lexicon_.count(ch)) continue;
      auto sub = ConvertWordToIds(ch); ans.insert(ans.end(), sub.begin(), sub.end());
    }
    return ans;
  }
#ifdef HAVE_ESPEAK
  std::vector<std::string> phs; english_to_codepoints(w, phs);
  for (const auto& ph : phs) { auto it = token2id_.find(ph); if (it != token2id_.end()) ans.push_back(it->second); }
#endif
  return ans;
}

std::vector<int32_t> MatchaFrontend::TextToIds(const std::string& text, int* out_skipped) const {
  std::string norm = normalize_text(text);
  std::vector<std::string> words = remove_spaces_after_punct(split_utf8(norm));
  std::vector<std::string> phrases = phrase_match(all_words_, words, 10);

  int blank = 1; { auto it = token2id_.find(" "); if (it != token2id_.end()) blank = it->second; }
  std::vector<int32_t> ids; int skipped = 0; std::string last_word;
  for (const std::string& w : phrases) {
    std::vector<int32_t> wids = ConvertWordToIds(w);
    if (wids.empty()) { skipped++; last_word = w; continue; }
    if (!last_word.empty() && std::isalpha((unsigned char)last_word[0])) ids.push_back(blank);
    ids.insert(ids.end(), wids.begin(), wids.end());
    last_word = w;
  }
  if (out_skipped) *out_skipped = skipped;
  return ids;
}
