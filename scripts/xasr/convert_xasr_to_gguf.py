#!/usr/bin/env python3
"""Convert the X-ASR zh-en streaming **zipformer2 transducer** (sherpa-onnx ONNX
release: encoder/decoder/joiner) into a single GGUF for RapidSpeech.cpp's
`XAsrZipformer2Model`.

Why ONNX-trace and not the .pt: the published model ships only ONNX. In the
encoder ONNX, Linear *weights* are anonymous `onnx::MatMul_NNNNN` initializers
(PyTorch fused Linear -> MatMul+Add); only the *.bias keeps a semantic name.
We recover each weight by tracing  bias -> consuming Add -> sibling MatMul ->
that MatMul's initializer input. Conv/embedding/norm/bypass weights are already
named. Decoder/joiner are fully named (Gemm); we honour transB.

Layout convention (identical to scripts/convert_qwen3_asr_to_gguf.py::_store):
PyTorch C-order bytes kept; declared shape reversed for ndim>=2 so ggml sees ne0
as the fastest axis. ONNX MatMul weight is [in,out] = PyTorch.T, so we transpose
to [out,in] before _store, making it bit-identical to a PyTorch Linear weight.

Usage:
  python tools/convert_xasr_to_gguf.py \
      --model-dir models/chunk-960ms \
      --encoder encoder-960ms.onnx --decoder decoder-960ms.onnx \
      --joiner joiner-960ms.onnx --tokens tokens.txt \
      --output models/x-asr-zh-en-960ms-f16.gguf [--f16]
"""
import argparse, os, sys
import numpy as np
import onnx
from onnx import numpy_helper

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")

ARCH = "XAsrZipformer2"


# ---------------------------------------------------------------- gguf storage
def _store(writer, name, t, use_f16):
    """Match convert_qwen3_asr_to_gguf.py::_store exactly."""
    t = np.asarray(t)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim >= 2:
        t = t.ravel().reshape(tuple(reversed(t.shape)))
    eff_ndim = sum(1 for d in t.shape if d > 1)
    if (use_f16 and eff_ndim >= 2 and not name.endswith(".bias")
            and not name.endswith("norm.weight") and not name.endswith(".log_scale")
            and not name.endswith(".bypass_scale")):
        t = t.astype(np.float16)
    else:
        t = t.astype(np.float32)
    writer.add_tensor(name, t)


# ---------------------------------------------------------------- onnx helpers
class Graph:
    def __init__(self, path):
        self.m = onnx.load(path, load_external_data=False)
        self.g = self.m.graph
        self.init = {t.name: t for t in self.g.initializer}
        self.meta = {p.key: p.value for p in self.m.metadata_props}
        self.producer = {}
        self.consumers = {}
        self.consumer_idx = {}  # value -> first consuming node index
        for idx, n in enumerate(self.g.node):
            for o in n.output:
                self.producer[o] = n
            for i in n.input:
                self.consumers.setdefault(i, []).append(n)
                self.consumer_idx.setdefault(i, idx)
        # node index where each named layer tensor is consumed (for orphan placement)
        self.named_at = sorted(
            (idx, i)
            for idx, n in enumerate(self.g.node)
            for i in n.input
            if i in self.init and not i.startswith("onnx::") and "layers" in i)

    def nearest_named(self, node_idx, suffix):
        best, bd = None, 1 << 30
        for idx, name in self.named_at:
            if name.endswith(suffix) and abs(idx - node_idx) < bd:
                bd, best = abs(idx - node_idx), name
        return best

    def arr(self, name):
        return numpy_helper.to_array(self.init[name])

    def trace_linear_weight(self, bias_name):
        """bias -> Add/Sub consumer -> sibling MatMul -> weight initializer (in,out)."""
        add = next((n for n in self.consumers.get(bias_name, [])
                    if n.op_type in ("Add", "Sub")), None)
        if add is None:
            return None
        for i in add.input:
            if i == bias_name:
                continue
            p = self.producer.get(i)
            if p and p.op_type == "MatMul":
                w = next((x for x in p.input if x in self.init), None)
                if w is not None:
                    return w, self.arr(w)  # (weight_name, shape (in, out))
        return None

    def gemm_weight(self, weight_name):
        """Return weight as PyTorch [out,in] regardless of the Gemm transB flag."""
        node = next((n for n in self.g.node
                     if n.op_type == "Gemm" and weight_name in n.input), None)
        w = self.arr(weight_name)
        transB = 0
        if node is not None:
            for a in node.attribute:
                if a.name == "transB":
                    transB = a.i
        # Gemm computes A*Bᵀ if transB else A*B. PyTorch Linear weight is [out,in]
        # and is exported with transB=1. If transB=0 the stored B is [in,out].
        return w if transB else w.T


def il(s):
    return [int(x) for x in str(s).split(",")]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--joiner", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--f16", action="store_true")
    a = ap.parse_args()
    d = a.model_dir
    enc = Graph(os.path.join(d, a.encoder))
    dec = Graph(os.path.join(d, a.decoder))
    joi = Graph(os.path.join(d, a.joiner))

    em = enc.meta
    num_layers = il(em["num_encoder_layers"])
    dims = il(em["encoder_dims"])
    kernels = il(em["cnn_module_kernels"])
    left_ctx = il(em["left_context_len"])
    qdims = il(em["query_head_dims"])
    vdims = il(em["value_head_dims"])
    nheads = il(em["num_heads"])
    n_stacks = len(num_layers)
    vocab = int(dec.meta["vocab_size"])
    context_size = int(dec.meta["context_size"])
    joiner_dim = int(joi.meta["joiner_dim"])

    tokens = []
    with open(os.path.join(d, a.tokens), encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split(" ")
            if len(parts) >= 1:
                tokens.append(parts[0])

    w = GGUFWriter(a.output, ARCH)  # sets general.architecture
    w.add_string("general.name", "X-ASR-zh-en streaming zipformer2")
    # --- encoder hparams (lists stored as int32 arrays) ---
    w.add_int32("xasr.encoder.n_stacks", n_stacks)
    w.add_array("xasr.encoder.num_layers", num_layers)
    w.add_array("xasr.encoder.dims", dims)
    w.add_array("xasr.encoder.cnn_kernels", kernels)
    w.add_array("xasr.encoder.left_context", left_ctx)
    w.add_array("xasr.encoder.query_head_dim", qdims)
    w.add_array("xasr.encoder.value_head_dim", vdims)
    w.add_array("xasr.encoder.num_heads", nheads)
    w.add_int32("xasr.encoder.decode_chunk_len", int(em["decode_chunk_len"]))
    w.add_int32("xasr.encoder.T", int(em["T"]))
    w.add_int32("xasr.encoder.feat_dim", 80)
    w.add_int32("xasr.encoder.out_dim", joiner_dim)
    # --- decoder / joiner / tokenizer ---
    w.add_int32("xasr.decoder.context_size", context_size)
    w.add_int32("xasr.joiner.dim", joiner_dim)
    w.add_int32("xasr.vocab_size", vocab)
    w.add_int32("tokenizer.vocab_size", len(tokens))
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_int32("xasr.sample_rate", 16000)

    n_w = 0
    orphan_mm = set(n for n in enc.init if n.startswith("onnx::MatMul"))

    # ---- encoder: emit named inits + traced Linear weights ----
    for t in enc.g.initializer:
        name = t.name
        if name.startswith("onnx::"):
            continue  # anonymous; emitted via tracing below
        arr = enc.arr(name)
        _store(w, name, arr, a.f16); n_w += 1
        if name.endswith(".bias"):
            traced = enc.trace_linear_weight(name)
            if traced is not None:
                src_name, wt = traced
                wname = name[:-len(".bias")] + ".weight"
                _store(w, wname, wt.T, a.f16); n_w += 1  # (in,out)->(out,in)
                orphan_mm.discard(src_name)

    # ---- encoder: map remaining anonymous weights (bias-less linear_pos) ----
    # Each is the relative-position projection of a layer's self_attn_weights;
    # assign by node-index proximity to that layer's named in_proj.bias.
    for src_name in sorted(orphan_mm):
        cidx = enc.consumer_idx[src_name]
        near = enc.nearest_named(cidx, "self_attn_weights.in_proj.bias")
        if near is None:
            continue
        prefix = near[:-len(".in_proj.bias")]  # ...self_attn_weights
        wt = enc.arr(src_name)  # (in,out)
        _store(w, prefix + ".linear_pos.weight", wt.T, a.f16); n_w += 1
        orphan_mm.discard(src_name)

    # ---- decoder: fully named ----
    _store(w, "decoder.embedding.weight", dec.arr("decoder.embedding.weight"), a.f16); n_w += 1
    _store(w, "decoder.conv.weight", dec.arr("decoder.conv.weight"), a.f16); n_w += 1
    _store(w, "decoder_proj.weight", dec.gemm_weight("decoder_proj.weight"), a.f16); n_w += 1
    _store(w, "decoder_proj.bias", dec.arr("decoder_proj.bias"), a.f16); n_w += 1

    # ---- joiner: fully named ----
    _store(w, "joiner.output_linear.weight", joi.gemm_weight("output_linear.weight"), a.f16); n_w += 1
    _store(w, "joiner.output_linear.bias", joi.arr("output_linear.bias"), a.f16); n_w += 1

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print(f"wrote {a.output}")
    print(f"  arch={ARCH} stacks={n_stacks} layers={num_layers} dims={dims}")
    print(f"  vocab={vocab} context_size={context_size} joiner_dim={joiner_dim} tokens={len(tokens)}")
    print(f"  tensors emitted: {n_w}")
    if orphan_mm:
        print(f"  WARNING: {len(orphan_mm)} anonymous MatMul weights NOT mapped "
              f"(first few: {sorted(orphan_mm)[:5]})", file=sys.stderr)
    else:
        print("  all anonymous encoder MatMul weights mapped ✓")


if __name__ == "__main__":
    main()
