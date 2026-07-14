#!/usr/bin/env bash
# Vendored ffmpeg.wasm single-thread core (ffmpeg-core.js + ffmpeg-core.wasm,
# ~32MB total). These are upstream build artifacts, gitignored like the model
# weights (see repo .gitignore: *.onnx / *.bin) rather than committed — this
# script re-fetches the exact pinned versions for a deploy. The small ESM
# wrapper files one level up (index.js, classes.js, worker.js, ...) ARE
# committed, since they're what app.js imports and are worth reviewing.
#
# Pinned to match the wrapper: @ffmpeg/ffmpeg@0.12.15 expects @ffmpeg/core
# 0.12.x. The single-thread core is used deliberately — it needs no
# SharedArrayBuffer, so it works regardless of cross-origin isolation.
set -euo pipefail
cd "$(dirname "$0")"
BASE="https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm"
curl -fSL -o ffmpeg-core.js   "$BASE/ffmpeg-core.js"
curl -fSL -o ffmpeg-core.wasm "$BASE/ffmpeg-core.wasm"
echo "fetched $(du -h ffmpeg-core.wasm | cut -f1) ffmpeg-core.wasm + ffmpeg-core.js"
