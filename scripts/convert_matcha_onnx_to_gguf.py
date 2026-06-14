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


def semantic_rename_map(graph, prefix):
    """ONNX exporters fold Linear/MatMul weights into anonymous `onnx::MatMul_*`
    initializers. Recover a semantic name from the consuming node's path, e.g.
    node `/blocks.0/pw1/MatMul` consuming `onnx::MatMul_284` -> `<prefix>.blocks.0.pw1.weight`.

    Some weights are first consumed by an `Identity` passthrough (e.g. when the 3 unrolled
    CFM-ODE copies share one weight) — naming by Identity gives a useless name, so we
    follow Identity outputs to the first *real* consumer (e.g. attn1/to_q/MatMul).
    Returns {init_name: semantic_name} for every anonymous initializer."""
    inits = {i.name for i in graph.initializer}
    # consumers[tensor] = list of nodes that read it
    consumers = {}
    for nd in graph.node:
        for inp in nd.input:
            consumers.setdefault(inp, []).append(nd)

    def real_consumer(tensor, depth=0):
        # first non-Identity consumer (follow Identity chains); returns (node, input_index, op)
        for nd in consumers.get(tensor, []):
            if nd.op_type == "Identity" and depth < 8:
                r = real_consumer(nd.output[0], depth + 1)
                if r is not None:
                    return r
            elif nd.op_type != "Identity":
                return (nd, list(nd.input).index(tensor), nd.op_type)
        return None

    ren = {}
    for inp in inits:
        if not inp.startswith("onnx::"):
            continue
        rc = real_consumer(inp)
        if rc is None:
            continue
        nd, idx, op = rc
        path = nd.name.strip("/").split("/")
        dropped = None
        if path and path[-1] in ("MatMul", "Gemm", "Conv", "Add", "Mul", "InstanceNormalization"):
            dropped = path[-1]; path = path[:-1]
        if not path:
            continue
        # Disambiguate only where two learned weights land on the SAME path:
        #  - InstanceNorm(scale=in1, bias=in2) -> .weight / .bias (decoder GroupNorm).
        #  - decoder GroupNorm affine: gamma is consumed by a leaf "Mul" node and beta by a leaf
        #    "Add" node, both children of the same `block.1` path -> collide; the dropped "Add"
        #    leaf -> .bias, otherwise .weight.
        # The encoder LayerNorm beta is consumed by node "Add_1" (a leaf NOT in the drop list, so
        # it stays in the path) -> no collision, keeps .weight (matches the validated encoder).
        if op == "InstanceNormalization":
            suffix = "weight" if idx == 1 else ("bias" if idx == 2 else f"in{idx}")
        elif dropped == "Add":
            suffix = "bias"
        else:
            suffix = "weight"
        ren[inp] = f"{prefix}.{'.'.join(path)}.{suffix}"
    return ren


def folded_constants(graph, prefix):
    """ONNX export folds some learned params (encoder LayerNorm gamma/beta, relative-
    position embeddings, Snake alpha/beta sometimes) into multi-element `Constant`
    nodes rather than initializers. Capture them keyed by the consuming node's path so
    the gguf carries every learned tensor. Skip 1-element scalars (loop/shape consts)."""
    out = {}
    consumers = {}  # const-output-name -> consuming node
    for nd in graph.node:
        for inp in nd.input:
            consumers.setdefault(inp, nd)
    for nd in graph.node:
        if nd.op_type != "Constant" or not nd.attribute:
            continue
        try:
            arr = numpy_helper.to_array(nd.attribute[0].t)
        except Exception:
            continue
        if arr.size <= 1:
            continue
        oc = nd.output[0]
        cons = consumers.get(oc)
        path = (cons.name.strip("/").replace("/", ".") if cons and cons.name else oc)
        out[f"{prefix}.const.{path}"] = arr
    return out


def load_inits(path, prefix):
    m = onnx.load(path)
    meta = {kv.key: kv.value for kv in m.metadata_props}
    ren = semantic_rename_map(m.graph, prefix)
    tensors = {}
    for i in m.graph.initializer:
        name = ren.get(i.name, i.name)
        tensors[name] = numpy_helper.to_array(i)
    # also capture folded multi-element Constants (encoder norms / rel-pos / etc.)
    nconst = 0
    for name, arr in folded_constants(m.graph, prefix).items():
        if name not in tensors:
            tensors[name] = arr
            nconst += 1
    return tensors, meta, len(ren), nconst


def main(src_dir, dst):
    acoustic = sorted(glob.glob(os.path.join(src_dir, "model-steps-*.onnx")))
    vocos = sorted(glob.glob(os.path.join(src_dir, "vocos-*.onnx")))
    if not acoustic or not vocos:
        sys.exit(f"need model-steps-*.onnx and vocos-*.onnx in {src_dir}")
    a_t, meta, a_ren, a_const = load_inits(acoustic[0], "model")
    v_t, _, v_ren, v_const = load_inits(vocos[0], "voc")
    print(f"semantic-renamed anonymous weights: acoustic {a_ren}, vocos {v_ren}")
    print(f"captured folded Constants (norms/rel-pos/etc): acoustic {a_const}, vocos {v_const}")
    # acoustic weights namespaced `model.*`, vocos `voc.*` — disjoint.
    tensors = {}
    tensors.update(a_t)
    tensors.update(v_t)

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
