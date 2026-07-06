from gguf import GGUFReader, GGUFWriter, GGUFValueType
srcs=['moss_nano.gguf','moss_codec.gguf']
w=GGUFWriter('moss_nano_full.gguf','moss_tts_nano')
seen=set()
for s in srcs:
    r=GGUFReader(s)
    for name,f in r.fields.items():
        if name in seen or name.startswith('GGUF') or name in ('general.architecture','general.name'): continue
        seen.add(name)
        vt=f.types[-1] if f.types else None
        try:
            if vt==GGUFValueType.STRING:
                raw=f.parts[f.data[0]]; w.add_string(name, bytes(raw).decode('utf-8'))
            elif vt==GGUFValueType.FLOAT32:
                w.add_float32(name, float(f.parts[f.data[0]][0]))
            elif vt in (GGUFValueType.UINT32,GGUFValueType.INT32,GGUFValueType.UINT64,GGUFValueType.INT64,GGUFValueType.UINT8,GGUFValueType.BOOL):
                w.add_uint32(name, int(f.parts[f.data[0]][0]))
            else:
                w.add_uint32(name, int(f.parts[f.data[0]][0]))
        except Exception as e:
            print('KV skip',name,vt,e)
    for t in r.tensors:
        w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("merged")
