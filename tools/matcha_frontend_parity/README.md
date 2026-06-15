# Matcha frontend parity test

Proves `frontend/matcha_frontend.{h,cpp}` is **byte-identical** to sherpa-onnx's matcha
text frontend (`matcha-tts-lexicon.cc` + `phrase-matcher` + `text-utils` + piper
`ProcessPhonemes`). Currently 222/222 cases pass (curated polyphones / mixed zh-en /
punctuation + 200 auto-generated lines).

## Pieces
- `frontend_dump.cpp` — links `matcha_frontend.cpp`, prints `IDS <ids>` for each arg text.
- `sherpa_gold.py` — runs `sherpa_onnx.OfflineTts` (matcha, `debug=True`) and extracts the
  gold token-id sequence (concatenated over its per-sentence split) from the debug log.
- `corpus.txt` — curated test sentences.

## Run
Build the dumper against espeak-ng **1.52** (the version the model was built with; 1.51
differs on English stress marks) and the model's `tokens.txt` + `lexicon.txt`:

```sh
g++ -std=c++17 -DHAVE_ESPEAK -I rapidspeech/src -I include -I ggml/include -I <espeak-inc> \
    frontend_dump.cpp ../../rapidspeech/src/frontend/matcha_frontend.cpp \
    ../../rapidspeech/src/utils/rs_log.cpp -o /tmp/fe <espeak-1.52.so>

# gold:
ESPEAK_DATA_PATH=<espeak-1.52-data> python3 sherpa_gold.py "你好，我是 Amy。"
# mine:
ESPEAK_DATA_PATH=<espeak-1.52-data> /tmp/fe tokens.txt lexicon.txt "你好，我是 Amy。"
```

Compare the id sequences; they must match exactly.
