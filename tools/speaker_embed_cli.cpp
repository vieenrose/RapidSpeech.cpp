// Batch speaker-embedding CLI over the rs_speaker C API (CAM++ gguf) — lets
// host processes (e.g. the native HF Space server) compute the SAME speaker
// embeddings as the WASM demo's rs_wasm_speaker_embed for cross-window
// diarization linking.
//
// Usage: rs-speaker-embed <campplus.gguf> <pcm_f32le_16k.raw> [a:b ...]
//   Each a:b is a sample range. Prints one line per range: space-separated
//   floats (dim from the model), or "nil" if the slice is too short / embed
//   fails. With no ranges, embeds the whole file.
#include "rapidspeech.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <campplus.gguf> <pcm_f32le.raw> [a:b ...]\n",
            argv[0]);
    return 2;
  }
  FILE *f = fopen(argv[2], "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
  fseek(f, 0, SEEK_END);
  long bytes = ftell(f);
  fseek(f, 0, SEEK_SET);
  std::vector<float> pcm(bytes / sizeof(float));
  if (fread(pcm.data(), 1, pcm.size() * sizeof(float), f) !=
      pcm.size() * sizeof(float)) {
    fclose(f);
    fprintf(stderr, "short read\n");
    return 1;
  }
  fclose(f);

  rs_speaker_t *sp = rs_speaker_init_from_file(argv[1], 4, /*use_gpu=*/false);
  if (!sp) { fprintf(stderr, "speaker init failed\n"); return 1; }
  const int dim = (int)rs_speaker_dim(sp);
  std::vector<float> emb(dim);

  std::vector<std::pair<long, long>> ranges;
  for (int i = 3; i < argc; ++i) {
    long a = 0, b = 0;
    if (sscanf(argv[i], "%ld:%ld", &a, &b) == 2) ranges.push_back({a, b});
  }
  if (ranges.empty()) ranges.push_back({0, (long)pcm.size()});

  for (auto [a, b] : ranges) {
    if (a < 0) a = 0;
    if (b > (long)pcm.size()) b = pcm.size();
    if (b - a < 16000 ||
        rs_speaker_embed(sp, pcm.data() + a, (int)(b - a), emb.data(), dim) !=
            RS_OK) {
      printf("nil\n");
      continue;
    }
    for (int d = 0; d < dim; ++d)
      printf(d ? " %.6g" : "%.6g", emb[d]);
    printf("\n");
  }
  rs_speaker_free(sp);
  return 0;
}
