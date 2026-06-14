#!/usr/bin/env python3
"""Convert a sherpa-onnx Matcha-TTS bundle (acoustic + vocos vocoder) to a single gguf
for the RapidSpeech.cpp ggml backend (Jetson Nano gen1 / CUDA 10.2 target).

Matcha-TTS = text encoder + duration predictor + conditional-flow-matching (CFM) decoder
(a UNet, InstanceNorm + Snake/Softplus activations, Sin time-embedding, N unrolled ODE
Euler steps), followed by a Vocos vocoder (ConvNeXt blocks + iSTFT head).

This script extracts every weight tensor from both ONNX graphs and writes them, keyed by
their original ONNX names (e.g. `model.encoder.emb.weight`, `voc.conv_pre.weight`), plus
the bundle's hyper-parameters (from the acoustic model's metadata_props) into gguf KV.

The consuming arch (rapidspeech/src/arch/matcha.cpp) builds the forward graph from these.

Usage:
  python convert_matcha_onnx_to_gguf.py <matcha-dir> <out.gguf>
    <matcha-dir> must contain model-steps-*.onnx and vocos-*.onnx

Validation: round-trips every tensor (gguf bytes == onnx numpy) before exit.
"""
import sys, os, glob
import numpy as np
import onnx
from onnx import numpy_helper
import gguf

ARCH = "matcha-tts"


def load_inits(path):
    m = onnx.load(path)
    meta = {kv.key: kv.value for kv in m.metadata_props}
    tensors = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    return tensors, meta


def main(src_dir, dst):
    acoustic = sorted(glob.glob(os.path.join(src_dir, "model-steps-*.onnx")))
    vocos = sorted(glob.glob(os.path.join(src_dir, "vocos-*.onnx")))
    if not acoustic or not vocos:
        sys.exit(f"need model-steps-*.onnx and vocos-*.onnx in {src_dir}")
    a_t, meta = load_inits(acoustic[0])
    v_t, _ = load_inits(vocos[0])
    # vocos weights are namespaced `voc.*`; acoustic `model.*` — disjoint, but prefix
    # the anonymous onnx::* constants per-graph to avoid collisions.
    def norm(name, ns):
        return name if (name.startswith("model.") or name.startswith("voc.")) else f"{ns}.{name}"
    tensors = {}
    for n, arr in a_t.items():
        tensors[norm(n, "acoustic")] = arr
    for n, arr in v_t.items():
        tensors[norm(n, "vocos")] = arr

    w = gguf.GGUFWriter(dst, ARCH)
    # hyper-parameters from the acoustic model card
    def mi(k, default=0):
        try: return int(meta.get(k, default))
        except Exception: return default
    w.add_uint32("matcha.sample_rate", mi("sample_rate", 8000))
    w.add_uint32("matcha.num_ode_steps", mi("num_ode_steps", 3))
    w.add_uint32("matcha.pad_id", mi("pad_id", 0))
    w.add_uint32("matcha.n_speakers", mi("n_speakers", 1))
    w.add_uint32("matcha.use_eos_bos", mi("use_eos_bos", 1))
    w.add_string("matcha.language", meta.get("language", ""))
    w.add_string("matcha.voice", meta.get("voice", ""))
    w.add_uint32("matcha.n_vocab", int(a_t["model.encoder.emb.weight"].shape[0]))
    w.add_uint32("matcha.hidden", int(a_t["model.encoder.emb.weight"].shape[1]))

    # tensors — gguf stores row-major; keep ONNX layout, the C++ loader knows each shape.
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr.astype(np.float32))
        w.add_tensor(name, arr)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {dst}: {len(tensors)} tensors "
          f"(acoustic {len(a_t)} + vocos {len(v_t)}), "
          f"sr={mi('sample_rate',8000)} ode_steps={mi('num_ode_steps',3)}")

    # round-trip check: re-read gguf, compare bytes to source
    # gguf records dims in reversed (ggml ne[]) order, so compare the flat row-major
    # buffer — that's the byte layout the ggml C++ loader reads back.
    r = gguf.GGUFReader(dst)
    rt = {t.name: t for t in r.tensors}
    bad = 0
    for name, arr in tensors.items():
        got = np.asarray(rt[name].data).reshape(-1).astype(np.float32)
        ref = np.ascontiguousarray(arr.astype(np.float32)).reshape(-1)
        if got.size != ref.size or not np.array_equal(got, ref):
            bad += 1
            if bad <= 5: print("  MISMATCH", name)
    print("round-trip:", "PASS" if bad == 0 else f"FAIL ({bad} tensors)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
