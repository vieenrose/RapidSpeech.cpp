#!/usr/bin/env python3
"""Inspect an X-ASR (zipformer2 streaming transducer) ONNX file:
metadata_props (the architecture spec), graph I/O (incl. streaming caches),
and a summary of initializer (weight) tensors by name prefix.
"""
import sys, onnx
from onnx import numpy_helper

def dt(t):  # elem type name
    return onnx.TensorProto.DataType.Name(t)

def shape(vi):
    d = vi.type.tensor_type.shape.dim
    return [ (x.dim_param or x.dim_value) for x in d ]

def main(path):
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    print(f"### {path}")
    print(f"ir_version={m.ir_version} producer={m.producer_name} opset={[ (o.domain or 'ai.onnx')+':'+str(o.version) for o in m.opset_import]}")
    print("\n=== metadata_props (architecture spec) ===")
    for p in m.metadata_props:
        print(f"  {p.key} = {p.value}")
    print(f"\n=== inputs ({len(g.input)}) ===")
    for vi in g.input:
        print(f"  {vi.name:40s} {dt(vi.type.tensor_type.elem_type):8s} {shape(vi)}")
    print(f"\n=== outputs ({len(g.output)}) ===")
    for vi in g.output:
        print(f"  {vi.name:40s} {dt(vi.type.tensor_type.elem_type):8s} {shape(vi)}")
    # initializer summary
    inits = list(g.initializer)
    print(f"\n=== initializers: {len(inits)} tensors ===")
    import collections, re
    bytes_by_dtype = collections.Counter()
    for t in inits:
        bytes_by_dtype[dt(t.data_type)] += 1
    print("  count by dtype:", dict(bytes_by_dtype))
    # group by top-level prefix (strip trailing .N indices)
    groups = collections.Counter()
    for t in inits:
        key = re.sub(r"\.\d+", ".N", t.name)
        groups[key] += 1
    print(f"\n=== initializer name patterns ({len(groups)} unique) ===")
    for k, c in sorted(groups.items()):
        print(f"  [{c:3d}] {k}")
    # node op histogram
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"\n=== op_type histogram ({len(g.node)} nodes) ===")
    for k, c in sorted(ops.items(), key=lambda x:-x[1]):
        print(f"  {c:5d}  {k}")

if __name__ == "__main__":
    main(sys.argv[1])
