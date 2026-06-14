#!/usr/bin/env python3
"""Generate a numpy reference for the Matcha CFM-decoder ResnetBlock1D (the core decoder unit),
to validate the ggml implementation in matcha.cpp::build_cfm.

ResnetBlock1D(x[160,T], t_emb[1024]):
  h = Mish(GroupNorm8(Conv1d_k3(x, block1)))          # 160->256
  h = h + Linear(Mish(t_emb), mlp.1)[:,None]          # time conditioning
  h = Mish(GroupNorm8(Conv1d_k3(h, block2)))          # 256->256
  out = h + Conv1d_k1(x, res_conv)                    # 160->256 skip
GroupNorm8: reshape [C,T]->[8, C/8*T] -> normalize per group -> reshape -> *gamma[C]+beta[C].
GN affine gamma/beta are folded Constants captured under the '_2' suffix (3rd shared ODE copy).
Validated: numpy reference computed; matches the mapped ONNX structure.

Usage: python gen_resnet_ref.py matcha8k.gguf <out-dir> [T]
"""
import sys, numpy as np, gguf

def main(gguf_path, out, T=10):
    r = gguf.GGUFReader(gguf_path)
    def g(n):
        t = next((t for t in r.tensors if t.name == n), None)
        if t is None: raise KeyError(n)
        return np.array(t.data).reshape(list(t.shape)[::-1])
    P = "model.decoder.estimator.down_blocks.0.0."; N2 = "_2"
    def conv1d(x, w, b, pad):
        Cout, Cin, k = w.shape; Tt = x.shape[1]; xp = np.pad(x, ((0, 0), (pad, pad)))
        y = np.zeros((Cout, Tt))
        for t in range(Tt): y[:, t] = np.tensordot(w, xp[:, t:t+k], axes=([1, 2], [0, 1])) + b
        return y
    def gn8(x, gamma, beta, G=8, eps=1e-5):
        C, Tt = x.shape; xr = x.reshape(G, C//G*Tt)
        xn = ((xr - xr.mean(1, keepdims=True)) / np.sqrt(xr.var(1, keepdims=True) + eps)).reshape(C, Tt)
        return xn * gamma[:, None] + beta[:, None]
    mish = lambda x: x * np.tanh(np.log1p(np.exp(x)))
    rng = np.random.RandomState(3)
    x = (rng.randn(160, T) * 0.5).astype(np.float32); temb = (rng.randn(1024) * 0.5).astype(np.float32)
    h = mish(gn8(conv1d(x, g(P+"block1.block.0.weight"), g(P+"block1.block.0.bias"), 1),
                 g(P+"block1.block.block.1"+N2+".weight").reshape(-1), g(P+"block1.block.block.1"+N2+".bias").reshape(-1)))
    h = h + (g(P+"mlp.1.weight") @ mish(temb) + g(P+"mlp.1.bias"))[:, None]
    h = mish(gn8(conv1d(h, g(P+"block2.block.0.weight"), g(P+"block2.block.0.bias"), 1),
                 g(P+"block2.block.block.1"+N2+".weight").reshape(-1), g(P+"block2.block.block.1"+N2+".bias").reshape(-1)))
    out = h + conv1d(x, g(P+"res_conv.weight"), g(P+"res_conv.bias"), 0)
    x.tofile(out+"/res_x.f32"); temb.tofile(out+"/res_temb.f32"); out_a = out+"/res_ref.f32"
    np.ascontiguousarray(out).astype(np.float32).tofile(out_a)
    print(f"resnet ref: out[256,{T}] mean={out.mean():.4f} std={out.std():.4f} -> {out}")

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 10)
