#include "frontend/matcha_frontend.h"
#include "utils/rs_log.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <string>

#ifdef HAVE_ESPEAK
#include <espeak-ng/speak_lib.h>
#include <cstdlib>
#endif

namespace {

#ifdef HAVE_ESPEAK
// English -> espeak-ng IPA phonemes, split into individual UTF-8 codepoints (the
// matcha-icefall model tokenizes IPA per codepoint: diphthongs like /oʊ/ are o + ʊ,
// stress marks ˈ ˌ and length ː are their own tokens). Initialized once; the data
// dir is ESPEAK_DATA_PATH (or espeak's default, e.g. /usr/lib/.../espeak-ng-data).
bool espeak_ready() {
  static int state = -1;  // -1 untried, 0 failed, 1 ok
  if (state < 0) {
    const char* dp = std::getenv("ESPEAK_DATA_PATH");
    int sr = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, dp, 0);
    state = (sr > 0 && espeak_SetVoiceByName("en-us") == EE_OK) ? 1 : 0;
    if (!state) RS_LOG_ERR("matcha frontend: espeak_Initialize failed (data dir? set ESPEAK_DATA_PATH)");
  }
  return state == 1;
}

// IPA->model-token replacements applied to the joined espeak string BEFORE splitting
// into codepoints. Mirrors sherpa-onnx matcha-tts-lexicon ApplyReplacements (PR #2853):
// the matcha-icefall model tokenizes English diphthongs as single tokens (A=eɪ, I=aɪ,
// O=oʊ/əʊ, Y=ɔɪ, W=aʊ) and uses ʧ/ʤ/ɡ/ɹ/ɛ. Order matters (eɪ->A before e->ɛ).
const std::pair<const char*, const char*> kReplacements[] = {
  {"ɝ","ɜɹ"},{"ɚ","əɹ"},
  {"eɪ","A"},{"aɪ","I"},{"ɔɪ","Y"},{"oʊ","O"},{"əʊ","O"},{"aʊ","W"},
  {"tʃ","ʧ"},{"dʒ","ʤ"},
  {"ː",""},
  {"g","ɡ"},{"r","ɹ"},
  {"e","ɛ"},
};

void english_to_codepoints(const std::string& seg, std::vector<std::string>& out) {
  if (!espeak_ready()) return;
  // 1) collect the full espeak IPA string (clause by clause).
  std::string ipa;
  const char* text = seg.c_str();
  const void* tp = (const void*)text;
  while (tp) {
    const char* ph = espeak_TextToPhonemes(&tp, espeakCHARS_UTF8, espeakPHONEMES_IPA);
    if (ph) ipa += ph;
    if (!ph) break;
  }
  // 2) apply the diphthong/symbol replacements in order.
  for (const auto& r : kReplacements) {
    std::string from = r.first, to = r.second; size_t pos = 0;
    while ((pos = ipa.find(from, pos)) != std::string::npos) { ipa.replace(pos, from.size(), to); pos += to.size(); }
  }
  // 3) split into UTF-8 codepoints (the model's tokens).
  for (size_t i = 0; i < ipa.size(); ) {
    unsigned char c = (unsigned char)ipa[i];
    int len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
    out.emplace_back(ipa, i, len);
    i += len;
  }
}
#endif

}  // namespace

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

// Map CJK / full-width sentence punctuation to its half-width token (kept, not dropped):
// these give clause boundaries the duration predictor needs — without them, an English
// word embedded between Chinese (e.g. "我是 Amy，") loses its boundary and collapses.
const char* cjk_punct_token(uint32_t cp) {
  switch (cp) {
    case 0x3002: case 0xFF0E:            return ".";   // 。 ．
    case 0xFF0C: case 0x3001:            return ",";   // ， 、
    case 0xFF01:                         return "!";   // ！
    case 0xFF1F:                         return "?";   // ？
    case 0xFF1A:                         return ":";   // ：
    case 0xFF1B:                         return ";";   // ；
    case 0x2026:                         return "…";   // …
    case 0x2014:                         return "—";   // —
    default:                             return nullptr;
  }
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
        continue;
      }
#ifdef HAVE_ESPEAK
      // English -> espeak IPA phonemes -> per-codepoint token ids.
      std::vector<std::string> phs;
      english_to_codepoints(s, phs);
      int hit = 0;
      for (auto& ph : phs) {
        auto pit = token2id_.find(ph);
        if (pit != token2id_.end()) { ids.push_back(pit->second); hit++; }
      }
      if (hit == 0) { skipped++; RS_LOG_INFO("matcha frontend: English '%s' produced no tokens (espeak/data?)", s.c_str()); }
#else
      // English path needs espeak-ng (build with -DRS_MATCHA_ESPEAK=ON); skip for now.
      skipped++;
      RS_LOG_INFO("matcha frontend: English segment '%s' needs espeak (not built in); skipped", s.c_str());
#endif
    } else if (const char* pt = cjk_punct_token(cp)) {
      // CJK / full-width punctuation -> its half-width token (clause boundary).
      auto it = token2id_.find(pt);
      if (it != token2id_.end()) ids.push_back(it->second);
      else skipped++;
    } else {
      // Other boundary char: dropped.
      skipped++;
    }
  }
  if (out_skipped) *out_skipped = skipped;
  return ids;
}
