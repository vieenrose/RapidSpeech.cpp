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
    for n, o in zip(names, outs):
        if n in ("mag", "x", "y"):
            o.astype(np.float32).tofile(os.path.join(out_dir, f"vocos_ref_{n}.f32"))
    print(f"fixture: mel[1,80,{T}] + mag/x/y[1,257,{T}] -> {out_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
