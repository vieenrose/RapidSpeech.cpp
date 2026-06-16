#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

/**
 * Text Frontend for TTS: converts text to phoneme ID sequences.
 *
 * Supports:
 *   - Chinese: simplified pinyin-based G2P with tone marks
 *   - English: CMU dictionary subset + rule-based fallback
 *
 * The phoneme vocabulary is shared with the OpenVoice2/MeloTTS model.
 */
class TextFrontend {
public:
  TextFrontend();
  ~TextFrontend() = default;

  /// Load phoneme vocabulary from a file or use built-in defaults.
  /// vocab_path can be nullptr to use built-in vocab.
  bool Init(const char* vocab_path = nullptr);

  /// Initialize from a symbol list (e.g. loaded from GGUF metadata).
  /// Each string in the list becomes a phoneme with ID = its index.
  bool InitFromSymbols(const std::vector<std::string>& symbols);

  /// Load the model's English word→phoneme(+tone) lexicon (MeloTTS lexicon.txt:
  /// "<word> <ph1>..<phN> <tone1>..<toneN>"). Only ASCII (English) entries are kept;
  /// Chinese rows are handled by the pinyin G2P. Replaces the tiny built-in dict, so
  /// out-of-the-54-word names (e.g. "michael") get correct ARPABET + English tones
  /// instead of the toy letter-by-letter fallback (which yields garbage audio).
  bool LoadEnglishLexicon(const char* path);

  /// Convert text to phoneme IDs for the given language.
  /// language: "zh", "en" (default: "zh")
  /// Returns phoneme ID sequence suitable for the TTS text encoder.
  /// If add_blank=true (MeloTTS default), interleaves `_` (id=0) between every phoneme;
  /// out_lang_ids per-phoneme will be 0 for blank tokens, language_id for real phonemes.
  std::vector<int32_t> TextToPhonemeIds(const std::string& text,
                                         const std::string& language = "zh",
                                         std::vector<int32_t>* out_tones = nullptr,
                                         std::vector<int32_t>* out_lang_ids = nullptr,
                                         int language_id = 0,
                                         bool add_blank = true,
                                         std::vector<int32_t>* out_word2ph = nullptr) const;

  /// Get vocabulary size (for model validation)
  int VocabSize() const { return static_cast<int>(phoneme_to_id_.size()); }

private:
  // Phoneme → ID mapping
  std::unordered_map<std::string, int32_t> phoneme_to_id_;

  // Chinese: character → pinyin mapping (simplified)
  std::unordered_map<uint32_t, std::string> char_to_pinyin_;

  // English: word → phoneme sequence + per-phoneme tone ids. The built-in dict has no
  // tones; the loaded model lexicon (LoadEnglishLexicon) supplies both. English tones in
  // MeloTTS live in their own range (e.g. 7–9), distinct from the Chinese tones (0–6) —
  // defaulting English to tone 0 mis-renders it even with correct phonemes.
  std::unordered_map<std::string, std::vector<std::string>> en_lexicon_;
  std::unordered_map<std::string, std::vector<int32_t>> en_tones_;
  int32_t en_default_tone_ = 0;   // tone for OOV English (rule fallback); set from lexicon

  // Internal methods
  std::vector<std::string> ChineseG2P(const std::string& text,
                                std::vector<int32_t>* out_tones = nullptr,
                                std::vector<int32_t>* out_word2ph = nullptr) const;
  std::vector<std::string> EnglishG2P(const std::string& text,
                                std::vector<int32_t>* out_tones = nullptr) const;
  std::vector<std::string> EnglishRuleFallback(const std::string& word) const;

  void InitBuiltinVocab();
  void InitChinesePinyin();
  void InitEnglishLexicon();
};
