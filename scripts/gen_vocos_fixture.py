#!/usr/bin/env python3
"""Generate a validation fixture for the ggml Vocos port: a deterministic mel input
and the ONNX vocos reference outputs (mag/x/y). Used by tools/matcha_vocos_validate.cpp.

Usage: python gen_vocos_fixture.py <vocos.onnx> <out-dir> [T]
Writes: <out-dir>/vocos_in_mel.f32  (mel [1,80,T])
        <out-dir>/vocos_ref_{mag,x,y}.f32  ([1,257,T] each)
"""
import sys, os
import numpy as np
import onnxruntime as ort

def main(onnx_path, out_dir, T=40):
    os.makedirs(out_dir, exist_ok=True)
    mel = (np.random.RandomState(1234).randn(1, 80, T) * 0.5).astype(np.float32)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {"mels": mel})
    names = [o.name for o in sess.get_outputs()]
    mel.tofile(os.path.join(out_dir, "vocos_in_mel.f32"))
    d = {n: o for n, o in zip(names, outs)}
    for n in ("mag", "x", "y"):
        d[n].astype(np.float32).tofile(os.path.join(out_dir, f"vocos_ref_{n}.f32"))
    # reference iSTFT waveform (n_fft=512 hop=128 periodic-hann center) from mag/x/y,
    # so the C++ iSTFT in matcha_vocos_validate.cpp can be checked against it.
    B, NFFT, HOP = 257, 512, 128
    mag, x, y = d["mag"][0], d["x"][0], d["y"][0]      # [257,T]; x=cos(ph), y=sin(ph)
    S = mag * (x + 1j * y)
    n = np.arange(NFFT); win = 0.5 - 0.5 * np.cos(2 * np.pi * n / NFFT)
    L = (T - 1) * HOP + NFFT
    out = np.zeros(L); wsum = np.zeros(L)
    for f in range(T):
        fr = np.fft.irfft(S[:, f], n=NFFT).real * win
        out[f * HOP:f * HOP + NFFT] += fr
        wsum[f * HOP:f * HOP + NFFT] += win ** 2
    wsum[wsum < 1e-8] = 1.0
    out = (out / wsum)[NFFT // 2: L - NFFT // 2]
    out.astype(np.float32).tofile(os.path.join(out_dir, "vocos_ref_wav.f32"))
    print(f"fixture: mel[1,80,{T}] + mag/x/y[1,257,{T}] + ref_wav[{len(out)}] -> {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
