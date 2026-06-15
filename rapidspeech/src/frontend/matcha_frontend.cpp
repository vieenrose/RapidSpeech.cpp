#include "frontend/matcha_frontend.h"
#include "utils/rs_log.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

namespace {

// Decode one UTF-8 char starting at p (< end); advance p past it. Returns the
// codepoint and, via `bytes`, the raw UTF-8 substring (used as the lexicon key).
uint32_t utf8_next(const char*& p, const char* end, std::string& bytes) {
  const char* start = p;
  unsigned char c = (unsigned char)*p++;
  uint32_t cp; int extra;
  if (c < 0x80)            { cp = c;        extra = 0; }
  else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
  else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
  else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
  else { bytes.assign(start, p); return 0xFFFD; }  // invalid lead byte
  for (int i = 0; i < extra && p < end; i++) cp = (cp << 6) | ((unsigned char)*p++ & 0x3F);
  bytes.assign(start, p);
  return cp;
}

inline bool is_cjk(uint32_t cp) { return cp >= 0x4E00 && cp <= 0x9FFF; }
// The reference ascii class: [a-zA-Z0-9 ,.!?]
inline bool is_ascii_keep(uint32_t cp) {
  if (cp >= 0x80) return false;
  char c = (char)cp;
  return std::isalnum((unsigned char)c) || c == ' ' || c == ',' ||
         c == '.' || c == '!' || c == '?';
}

std::string strip(const std::string& s) {
  size_t a = s.find_first_not_of(" \t");
  if (a == std::string::npos) return "";
  size_t b = s.find_last_not_of(" \t");
  return s.substr(a, b - a + 1);
}

}  // namespace

bool MatchaFrontend::Load(const char* tokens_path, const char* lexicon_path) {
  if (!tokens_path || !lexicon_path) { RS_LOG_ERR("matcha frontend: null path"); return false; }
  token2id_.clear(); lexicon_.clear();

  std::ifstream tf(tokens_path);
  if (!tf) { RS_LOG_ERR("matcha frontend: cannot open tokens %s", tokens_path); return false; }
  std::string line;
  while (std::getline(tf, line)) {
    std::istringstream is(line);
    std::vector<std::string> f; std::string w;
    while (is >> w) f.push_back(w);
    if (f.empty()) continue;
    if (f.size() == 1) token2id_[" "] = std::stoi(f[0]);          // bare id => space token
    else               token2id_[f[0]] = std::stoi(f[1]);
  }
  if (token2id_.empty()) { RS_LOG_ERR("matcha frontend: empty tokens %s", tokens_path); return false; }

  std::ifstream lf(lexicon_path);
  if (!lf) { RS_LOG_ERR("matcha frontend: cannot open lexicon %s", lexicon_path); return false; }
  int missing_tok = 0;
  while (std::getline(lf, line)) {
    std::istringstream is(line);
    std::string word; if (!(is >> word)) continue;
    std::vector<int32_t> ids; std::string t; bool ok = true;
    while (is >> t) {
      auto it = token2id_.find(t);
      if (it == token2id_.end()) { ok = false; missing_tok++; break; }
      ids.push_back(it->second);
    }
    if (ok && !ids.empty()) lexicon_[word] = std::move(ids);
  }
  if (lexicon_.empty()) { RS_LOG_ERR("matcha frontend: empty lexicon %s", lexicon_path); return false; }
  RS_LOG_INFO("matcha frontend: %d tokens, %d lexicon entries%s",
              (int)token2id_.size(), (int)lexicon_.size(),
              missing_tok ? " (some lexicon lines dropped: unknown token)" : "");
  return true;
}

std::vector<int32_t> MatchaFrontend::TextToIds(const std::string& text, int* out_skipped) const {
  std::vector<int32_t> ids;
  int skipped = 0;
  const char* p = text.c_str();
  const char* end = p + text.size();

  while (p < end) {
    // Classify the next codepoint without consuming it for run grouping.
    const char* save = p; std::string bytes;
    uint32_t cp = utf8_next(p, end, bytes);

    if (is_cjk(cp)) {
      // CJK run: per-character lexicon lookup (faithful to test.py).
      p = save;
      while (p < end) {
        const char* s2 = p; std::string cb;
        uint32_t c2 = utf8_next(p, end, cb);
        if (!is_cjk(c2)) { p = s2; break; }
        auto it = lexicon_.find(cb);
        if (it != lexicon_.end()) ids.insert(ids.end(), it->second.begin(), it->second.end());
        else { skipped++; RS_LOG_INFO("matcha frontend: no lexicon entry for '%s' (skipped)", cb.c_str()); }
      }
    } else if (is_ascii_keep(cp)) {
      // ASCII run: collect, then either a whole-run token (punctuation) or English.
      p = save;
      std::string run;
      while (p < end) {
        const char* s2 = p; std::string cb;
        uint32_t c2 = utf8_next(p, end, cb);
        if (!is_ascii_keep(c2)) { p = s2; break; }
        run += cb;
      }
      std::string s = strip(run);
      if (s.empty()) continue;
      auto it = token2id_.find(s);
      if (it != token2id_.end()) {
        ids.push_back(it->second);                 // lone punctuation, e.g. "." "," "?"
      } else {
        // English -> espeak phonemes. Not yet wired (needs espeak-ng); skip.
        skipped++;
        RS_LOG_INFO("matcha frontend: English segment '%s' needs espeak (not wired); skipped", s.c_str());
      }
    } else {
      // Boundary char (full-width punctuation, etc.): dropped, as in the reference.
      skipped++;
    }
  }
  if (out_skipped) *out_skipped = skipped;
  return ids;
}
