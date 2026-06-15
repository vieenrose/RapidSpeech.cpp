#pragma once
// Text frontend for the Matcha-TTS (icefall zh-en) acoustic model.
//
// This is a SEPARATE vocabulary from MeloTTS/OpenVoice2 (see text_frontend.h):
// matcha-icefall-zh-en is driven by `tokens.txt` (token->id) + `lexicon.txt`
// (word -> space-separated tokens), exactly as sherpa-onnx's Matcha frontend and
// the reference scripts/matcha-tts/zh-en/test.py:
//
//   text  --regex-->  [ CJK run | ascii run ]
//     CJK run   : per-character lexicon lookup -> pinyin-with-tone tokens (e.g. 你->ni3)
//     ascii run : if the whole (stripped) run is a token (punctuation) -> that id;
//                 otherwise English -> espeak phonemes (piper_phonemize, "en-us").
//   token -> id via tokens.txt; ids are fed to MatchaModel as phoneme_ids.
//
// The Chinese path is fully self-contained (just the two data files; the lexicon
// covers both Simplified and Traditional, so zh-TW works with no conversion).
// The English path needs espeak-ng and is wired separately (HAVE_ESPEAK); without
// it, non-Chinese segments are skipped with a warning.
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MatchaFrontend {
public:
  MatchaFrontend() = default;

  /// Load tokens.txt (token<space>id per line; a line that is just an id maps the
  /// space token " ") and lexicon.txt (word followed by its space-separated tokens).
  /// Returns false if either file cannot be read.
  bool Load(const char* tokens_path, const char* lexicon_path);

  bool Ready() const { return !token2id_.empty() && !lexicon_.empty(); }

  /// Convert UTF-8 text to Matcha phoneme IDs. Unknown CJK characters and (when
  /// espeak is unavailable) English segments are skipped; `out_skipped`, if given,
  /// receives the count of skipped pieces.
  std::vector<int32_t> TextToIds(const std::string& text, int* out_skipped = nullptr) const;

  int VocabSize() const { return (int)token2id_.size(); }
  int LexiconSize() const { return (int)lexicon_.size(); }

private:
  // word (or phrase) -> token ids, recursing into CJK chars / espeak for English.
  std::vector<int32_t> ConvertWordToIds(const std::string& w) const;

  std::unordered_map<std::string, int32_t> token2id_;
  std::unordered_map<std::string, std::vector<int32_t>> lexicon_;  // word -> token ids (word2ids_)
  std::unordered_set<std::string> all_words_;                      // lexicon keys, for phrase matching
};
