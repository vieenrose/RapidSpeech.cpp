#!/usr/bin/env python3
"""Convert MOSS-TTS-Nano-100M (AR model) to GGUF for RapidSpeech.cpp / ggml-CUDA.

Backbone = GPT-2 (nn.Linear, RoPE, gelu_new). Two transformers: global (12L) +
local (1L). 16 audio codebook embeddings/heads (tied to embeddings), text head
tied to wte. Big 2D linear weights -> Q8_0 (bandwidth-bound mul_mat_vec on Maxwell,
Q8_0 ~= lossless + 2.4x faster); norms/biases stay F32.

Usage: python convert_moss_nano_to_gguf.py --src <dir with pytorch_model.bin,config.json> --out moss_nano.gguf
"""
import argparse, json
from pathlib import Path
import numpy as np, torch
from gguf import GGUFWriter, GGMLQuantizationType
from gguf.quants import quantize

Q8 = GGMLQuantizationType.Q8_0
F32 = GGMLQuantizationType.F32

# big 2D matmul weights -> Q8_0; everything else (ln weight/bias, biases) -> F32
def is_quant(name, arr):
    return arr.ndim == 2 and name.endswith(".weight") and arr.shape[-1] % 32 == 0

def add(w, name, t):
    arr = t.detach().to(torch.float32).numpy()
    if is_quant(name, arr):
        w.add_tensor(name, quantize(arr, Q8), raw_dtype=Q8)
        return "Q8_0", arr.shape
    w.add_tensor(name, arr, raw_dtype=F32)
    return "F32 ", arr.shape

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/home/luigi/moss-port")
    ap.add_argument("--out", default="/home/luigi/moss-port/moss_nano.gguf")
    a = ap.parse_args()
    src = Path(a.src)
    cfg = json.load(open(src / "config.json"))
    g = cfg["gpt2_config"]
    sd = torch.load(src / "pytorch_model.bin", map_location="cpu", weights_only=True)

    w = GGUFWriter(a.out, "moss_tts_nano")
    # ---- hparams ----
    w.add_uint32("moss.n_layer", g["n_layer"])
    w.add_uint32("moss.n_local_layer", int(cfg["local_transformer_layers"]))
    w.add_uint32("moss.n_embd", g["n_embd"])
    w.add_uint32("moss.n_head", g["n_head"])
    w.add_uint32("moss.n_ff", g["n_inner"])
    w.add_uint32("moss.head_dim", g["n_embd"] // g["n_head"])
    w.add_uint32("moss.vocab_size", g["vocab_size"])
    w.add_uint32("moss.n_codebooks", len(cfg["audio_codebook_sizes"]))
    w.add_uint32("moss.codebook_size", cfg["audio_vocab_size"])
    w.add_float32("moss.rope_base", float(g.get("rope_base", 10000.0)))
    w.add_uint32("moss.sample_rate", cfg["audio_tokenizer_sample_rate"])
    # special token ids used by the prompt/generation loop
    for k in ["audio_start_token_id","audio_end_token_id","audio_pad_token_id",
              "audio_user_slot_token_id","audio_assistant_slot_token_id"]:
        w.add_uint32("moss."+k, int(cfg[k]))

    # ---- tensor name maps ----
    def block_map(pfx_src, pfx_dst, n):
        for i in range(n):
            s, d = f"{pfx_src}.h.{i}", f"{pfx_dst}blk.{i}"
            yield f"{s}.ln_1.weight",     f"{d}.attn_norm.weight"
            yield f"{s}.ln_1.bias",       f"{d}.attn_norm.bias"
            yield f"{s}.attn.c_attn.weight", f"{d}.attn_qkv.weight"
            yield f"{s}.attn.c_attn.bias",   f"{d}.attn_qkv.bias"
            yield f"{s}.attn.c_proj.weight", f"{d}.attn_out.weight"
            yield f"{s}.attn.c_proj.bias",   f"{d}.attn_out.bias"
            yield f"{s}.ln_2.weight",     f"{d}.ffn_norm.weight"
            yield f"{s}.ln_2.bias",       f"{d}.ffn_norm.bias"
            yield f"{s}.mlp.fc_in.weight",  f"{d}.ffn_up.weight"
            yield f"{s}.mlp.fc_in.bias",    f"{d}.ffn_up.bias"
            yield f"{s}.mlp.fc_out.weight", f"{d}.ffn_down.weight"
            yield f"{s}.mlp.fc_out.bias",   f"{d}.ffn_down.bias"

    pairs = []
    pairs.append(("transformer.wte.weight", "token_embd.weight"))          # tied text_lm_head
    pairs += list(block_map("transformer", "", g["n_layer"]))
    pairs.append(("transformer.ln_f.weight", "output_norm.weight"))
    pairs.append(("transformer.ln_f.bias",   "output_norm.bias"))
    pairs += list(block_map("local_transformer", "local.", int(cfg["local_transformer_layers"])))
    pairs.append(("local_transformer.ln_f.weight", "local.output_norm.weight"))
    pairs.append(("local_transformer.ln_f.bias",   "local.output_norm.bias"))
    for k in range(len(cfg["audio_codebook_sizes"])):
        pairs.append((f"audio_embeddings.{k}.weight", f"audio_embd.{k}.weight"))  # tied audio_lm_heads

    nq = nf = 0
    for src_name, dst_name in pairs:
        if src_name not in sd:
            raise KeyError(f"missing {src_name}")
        kind, shp = add(w, dst_name, sd[src_name])
        if kind.startswith("Q8"): nq += 1
        else: nf += 1
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print(f"wrote {a.out}: {len(pairs)} tensors ({nq} Q8_0, {nf} F32)")

if __name__ == "__main__":
    main()
