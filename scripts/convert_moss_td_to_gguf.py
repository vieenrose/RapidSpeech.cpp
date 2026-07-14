#!/usr/bin/env python3
"""
Convert HuggingFace MOSS-Transcribe-Diarize
(MossTranscribeDiarizeForConditionalGeneration) to a single GGUF file
compatible with RapidSpeech.cpp's `MossTDModel` arch.

MOSS-TD = classic Whisper-medium audio encoder (80-mel, conv1d stem, 24 layers)
        + VQAdaptor (Linear 4096->1024 -> SiLU -> Linear 1024->1024 -> LayerNorm)
          fed by a contiguous 4-frame time-merge (1024*4 = 4096)
        + Qwen3-0.6B decoder (28 layers, GQA 16/8, head_dim 128, tied embeddings)
        + audio embeddings masked-scattered into the decoder token stream at
          audio_token_id (151671).

Output layout:
  general.architecture = "MossTD"
  mosstd.encoder.*  — Whisper audio tower hparams
  mosstd.mm.*       — adaptor / merge dims + audio_token_id
  qwen3.*           — LLM hparams (reuses the existing Qwen3 GGUF keys so the
                      existing llm_model Qwen3 loader/build can be shared)
  tokenizer.*       — Qwen3 BPE vocab + merges
  Tensors:
    a.conv1.{weight,bias}                 — whisper_encoder.conv1 (conv1d 80->1024 k3 s1)
    a.conv2.{weight,bias}                 — whisper_encoder.conv2 (conv1d 1024->1024 k3 s2)
    a.position_embd.weight                — whisper_encoder.embed_positions.weight [1500,1024]
    a.blk.<N>.attn_{q,k,v,o}.{weight,bias}   (k has no bias — Whisper)
    a.blk.<N>.attn_norm.{weight,bias}     — self_attn_layer_norm (pre-attn LN)
    a.blk.<N>.ffn_{up,down}.{weight,bias} — fc1 / fc2
    a.blk.<N>.ffn_norm.{weight,bias}      — final_layer_norm (pre-MLP LN)
    a.post_norm.{weight,bias}             — whisper_encoder.layer_norm (post encoder)
    mm.a.0.{weight,bias}                  — vq_adaptor.layers.0  Linear 4096->1024
    mm.a.2.{weight,bias}                  — vq_adaptor.layers.2  Linear 1024->1024
    mm.a.norm.{weight,bias}               — vq_adaptor.layers.3  LayerNorm(1024, eps 1e-6)
    llm.model.embed_tokens.weight, llm.model.layers.N.*, llm.model.norm.weight,
    llm.lm_head.weight (== embed_tokens, tied)   — language_model.*

Memory-layout convention follows convert_qwen3_asr_to_gguf.py: PyTorch weights
keep their native C-order buffer; only the *declared* shape is reversed
(`ravel().reshape(*reversed(shape))`) so ggml sees ne0 as the fastest axis.
conv1d weight [OC,IC,K] reversed -> ne=[K,IC,OC], which is exactly ggml_conv_1d's
kernel layout.

Usage:
  python scripts/convert_moss_td_to_gguf.py \
      --hf-dir /path/to/moss_ft_zhtw_v5_kl \
      --output models/moss-td-f16.gguf [--f16]

Requires: transformers, safetensors, numpy, gguf.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter
except ImportError:
    print("Please install gguf: pip install gguf", file=sys.stderr)
    raise


# ----------------------------------------------------------------------------
# Encoder tensor name mapping (HF whisper_encoder.* -> GGUF a.*)
# ----------------------------------------------------------------------------

def encoder_name_map(hf_name: str, n_enc_layers: int) -> str | None:
    n = hf_name
    P = "model.whisper_encoder."
    if not n.startswith(P):
        return None
    rest = n[len(P):]

    # Conv stem (conv1d)
    conv = {
        "conv1.weight": "a.conv1.weight", "conv1.bias": "a.conv1.bias",
        "conv2.weight": "a.conv2.weight", "conv2.bias": "a.conv2.bias",
        "embed_positions.weight": "a.position_embd.weight",
        "layer_norm.weight": "a.post_norm.weight",
        "layer_norm.bias":   "a.post_norm.bias",
    }
    if rest in conv:
        return conv[rest]

    # Transformer layers: whisper_encoder.layers.<N>.<suffix>
    if rest.startswith("layers."):
        idx_str, suffix = rest[len("layers."):].split(".", 1)
        try:
            il = int(idx_str)
        except ValueError:
            return None
        if il < 0 or il >= n_enc_layers:
            return None
        base = f"a.blk.{il}."
        mapping = {
            "self_attn.q_proj.weight":     base + "attn_q.weight",
            "self_attn.q_proj.bias":       base + "attn_q.bias",
            "self_attn.k_proj.weight":     base + "attn_k.weight",   # no bias (Whisper)
            "self_attn.v_proj.weight":     base + "attn_v.weight",
            "self_attn.v_proj.bias":       base + "attn_v.bias",
            "self_attn.out_proj.weight":   base + "attn_o.weight",
            "self_attn.out_proj.bias":     base + "attn_o.bias",
            "self_attn_layer_norm.weight": base + "attn_norm.weight",
            "self_attn_layer_norm.bias":   base + "attn_norm.bias",
            "fc1.weight":                  base + "ffn_up.weight",
            "fc1.bias":                    base + "ffn_up.bias",
            "fc2.weight":                  base + "ffn_down.weight",
            "fc2.bias":                    base + "ffn_down.bias",
            "final_layer_norm.weight":     base + "ffn_norm.weight",
            "final_layer_norm.bias":       base + "ffn_norm.bias",
        }
        return mapping.get(suffix)
    return None


def adaptor_name_map(hf_name: str) -> str | None:
    m = {
        "model.vq_adaptor.layers.0.weight": "mm.a.0.weight",
        "model.vq_adaptor.layers.0.bias":   "mm.a.0.bias",
        "model.vq_adaptor.layers.2.weight": "mm.a.2.weight",
        "model.vq_adaptor.layers.2.bias":   "mm.a.2.bias",
        "model.vq_adaptor.layers.3.weight": "mm.a.norm.weight",
        "model.vq_adaptor.layers.3.bias":   "mm.a.norm.bias",
    }
    return m.get(hf_name)


def llm_name_map(hf_name: str) -> str | None:
    """model.language_model.* -> llm.model.* (matches the Qwen3 llm loader)."""
    P = "model.language_model."
    if hf_name.startswith(P):
        return "llm.model." + hf_name[len(P):]
    return None


# ----------------------------------------------------------------------------
# Tensor write helper (reverse shape -> ggml ne0-fastest; keep 1-D / norms F32)
# ----------------------------------------------------------------------------

def _store(writer: GGUFWriter, name: str, t: np.ndarray, use_f16: bool):
    # gguf sets ggml ne = reversed(numpy.shape). PyTorch weights are already in
    # the C-order that makes that reversal land ne0 on the fastest (last PT)
    # axis — which is exactly what ggml_mul_mat (weights [in, out]) and
    # ggml_conv_1d (kernel [K, IC, OC], from PT [OC, IC, K]) expect. So pass
    # the array through directly; do NOT pre-reverse the shape.
    t = np.ascontiguousarray(t)
    eff_ndim = sum(1 for d in t.shape if d > 1)
    if use_f16 and eff_ndim >= 2 and not name.endswith(".bias") and \
            not name.endswith("norm.weight"):
        t = t.astype(np.float16)
    else:
        t = t.astype(np.float32)
    writer.add_tensor(name, t)


def convert(hf_dir: Path, output: Path, use_f16: bool):
    print(f"Loading from {hf_dir}")
    with open(hf_dir / "config.json") as f:
        cfg = json.load(f)

    audio_cfg = cfg.get("audio_config", {})
    text_cfg = cfg.get("text_config", {})

    # Encoder (Whisper) hparams
    n_mels       = int(audio_cfg.get("num_mel_bins", 80))
    enc_d_model  = int(audio_cfg.get("d_model", 1024))
    enc_n_head   = int(audio_cfg.get("encoder_attention_heads", 16))
    enc_n_layer  = int(audio_cfg.get("encoder_layers", 24))
    enc_ffn_dim  = int(audio_cfg.get("encoder_ffn_dim", 4096))
    enc_max_pos  = int(audio_cfg.get("max_source_positions", 1500))
    enc_norm_eps = 1e-5
    n_fft        = 400
    hop_length   = 160
    sample_rate  = 16000

    # Adaptor / merge
    merge_size     = int(cfg.get("audio_merge_size", 4))
    audio_token_id = int(cfg.get("audio_token_id", 151671))
    adaptor_in     = int(cfg.get("adaptor_input_dim", enc_d_model * merge_size))
    adaptor_eps    = float(text_cfg.get("rms_norm_eps", 1e-6))

    # LLM (Qwen3) hparams
    n_llm_layer   = int(text_cfg.get("num_hidden_layers", 28))
    n_llm_embd    = int(text_cfg.get("hidden_size", 1024))
    n_llm_head    = int(text_cfg.get("num_attention_heads", 16))
    n_llm_head_kv = int(text_cfg.get("num_key_value_heads", 8))
    n_llm_vocab   = int(text_cfg.get("vocab_size", 151936))
    n_llm_ff      = int(text_cfg.get("intermediate_size", 3072))
    head_dim      = int(text_cfg.get("head_dim", 128))
    rope_base     = float(text_cfg.get("rope_theta", 1000000.0))
    rms_eps       = float(text_cfg.get("rms_norm_eps", 1e-6))
    n_ctx_train   = int(text_cfg.get("max_position_embeddings", 131072))
    bos           = int(cfg.get("pad_token_id", 151643))
    eos           = 151645  # <|im_end|> (generation_config)

    print(f"Encoder: d_model={enc_d_model} n_head={enc_n_head} n_layer={enc_n_layer} "
          f"ffn={enc_ffn_dim} n_mels={n_mels}")
    print(f"Adaptor: in={adaptor_in} out={n_llm_embd} merge={merge_size} "
          f"audio_token_id={audio_token_id}")
    print(f"LLM:     n_layer={n_llm_layer} n_embd={n_llm_embd} n_head={n_llm_head}/"
          f"{n_llm_head_kv} head_dim={head_dim} vocab={n_llm_vocab}")

    writer = GGUFWriter(str(output), "MossTD")
    writer.add_string("general.architecture", "MossTD")
    writer.add_string("general.name", "MOSS-Transcribe-Diarize-zhTW")

    # Encoder
    writer.add_int32("mosstd.encoder.n_mels",     n_mels)
    writer.add_int32("mosstd.encoder.n_fft",      n_fft)
    writer.add_int32("mosstd.encoder.hop_length", hop_length)
    writer.add_int32("mosstd.encoder.sample_rate", sample_rate)
    writer.add_int32("mosstd.encoder.d_model",    enc_d_model)
    writer.add_int32("mosstd.encoder.n_head",     enc_n_head)
    writer.add_int32("mosstd.encoder.n_layer",    enc_n_layer)
    writer.add_int32("mosstd.encoder.ffn_dim",    enc_ffn_dim)
    writer.add_int32("mosstd.encoder.max_source_positions", enc_max_pos)
    writer.add_float32("mosstd.encoder.layer_norm_eps", enc_norm_eps)
    # Adaptor / merge
    writer.add_int32("mosstd.mm.merge_size",     merge_size)
    writer.add_int32("mosstd.mm.in_dim",         adaptor_in)
    writer.add_int32("mosstd.mm.out_dim",        n_llm_embd)
    writer.add_int32("mosstd.mm.audio_token_id", audio_token_id)
    writer.add_float32("mosstd.mm.norm_eps",     adaptor_eps)

    # LLM (Qwen3) — same keys llm_model::load_hparams consumes
    writer.add_int32("qwen3.block_count",                        n_llm_layer)
    writer.add_int32("qwen3.embedding_length",                   n_llm_embd)
    writer.add_int32("qwen3.attention.head_count",               n_llm_head)
    writer.add_int32("qwen3.attention.head_count_kv",            n_llm_head_kv)
    writer.add_int32("qwen3.attention.key_length",               head_dim)
    writer.add_int32("qwen3.attention.value_length",             head_dim)
    writer.add_float32("qwen3.attention.layer_norm_rms_epsilon", rms_eps)
    writer.add_int32("qwen3.feed_forward_length",                n_llm_ff)
    writer.add_int32("qwen3.context_length",                     n_ctx_train)
    writer.add_float32("qwen3.rope.freq_base",                   rope_base)
    writer.add_int32("qwen3.bos_token_id", bos)
    writer.add_int32("qwen3.eos_token_id", eos)

    # ---- Tokenizer -------------------------------------------------------
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("Please install transformers: pip install transformers")
    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    vocab_size = getattr(tok, "vocab_size", n_llm_vocab)

    id_to_token = {}
    for tok_text, tid in tok.get_vocab().items():
        id_to_token[int(tid)] = tok_text
    for tid, tt in tok.added_tokens_decoder.items():
        id_to_token[int(tid)] = tt.content if hasattr(tt, "content") else str(tt)
    max_id = max(id_to_token) if id_to_token else 0
    tokens = [id_to_token.get(i, "") for i in range(max(vocab_size, max_id + 1))]

    writer.add_int32("tokenizer.vocab_size", len(tokens))
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_string("tokenizer.unk_symbol", "<|endoftext|>")

    merges = _load_merges(hf_dir)
    if merges:
        writer.add_array("tokenizer.ggml.merges", merges)
        print(f"BPE merges: {len(merges)}")
    else:
        print("WARNING: no BPE merges found", file=sys.stderr)

    # ---- Tensors ---------------------------------------------------------
    state = _load_state_dict(hf_dir)
    n_enc = n_mm = n_llm = skipped = 0
    embed_np = None

    for name in sorted(state.keys()):
        import torch
        t = state[name].detach().to(dtype=torch.float32).cpu().numpy()

        g = encoder_name_map(name, enc_n_layer)
        if g is not None:
            _store(writer, g, t, use_f16); n_enc += 1; continue
        g = adaptor_name_map(name)
        if g is not None:
            _store(writer, g, t, use_f16); n_mm += 1; continue
        g = llm_name_map(name)
        if g is not None:
            _store(writer, g, t, use_f16); n_llm += 1
            if name == "model.language_model.embed_tokens.weight":
                embed_np = t
            continue
        skipped += 1

    # Tied lm_head = embed_tokens (MOSS has no separate lm_head tensor).
    if embed_np is not None:
        _store(writer, "llm.lm_head.weight", embed_np, use_f16); n_llm += 1
        print("Emitted tied llm.lm_head.weight = embed_tokens")
    else:
        print("WARNING: embed_tokens not found — lm_head missing", file=sys.stderr)

    print(f"Encoder tensors: {n_enc}   Adaptor tensors: {n_mm}   "
          f"LLM tensors: {n_llm}   Skipped: {skipped}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWritten {output}")


def _load_merges(hf_dir: Path):
    """BPE merges as 'a b' strings. Prefer merges.txt, else tokenizer.json
    (newer HF format stores them as [left, right] pair lists)."""
    mf = hf_dir / "merges.txt"
    if mf.exists():
        with open(mf, "r", encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f
                    if ln.strip() and not ln.startswith("#")]
    tj = hf_dir / "tokenizer.json"
    if tj.exists():
        with open(tj, encoding="utf-8") as f:
            data = json.load(f)
        raw = data.get("model", {}).get("merges", [])
        out = []
        for m in raw:
            if isinstance(m, (list, tuple)):
                out.append(f"{m[0]} {m[1]}")
            else:
                out.append(m)
        return out
    return []


def _load_state_dict(hf_dir: Path):
    import safetensors.torch
    index = hf_dir / "model.safetensors.index.json"
    single = hf_dir / "model.safetensors"
    if index.exists():
        with open(index) as f:
            idx = json.load(f)
        state = {}
        for fn in sorted(set(idx["weight_map"].values())):
            print(f"  Loading shard {fn}")
            state.update(safetensors.torch.load_file(str(hf_dir / fn), device="cpu"))
        return state
    if single.exists():
        print(f"  Loading {single.name}")
        return safetensors.torch.load_file(str(single), device="cpu")
    sys.exit(f"No safetensors weights found in {hf_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--f16", action="store_true", help="Store 2-D weights as F16")
    args = ap.parse_args()
    hf_dir = Path(args.hf_dir).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    convert(hf_dir, output, args.f16)
