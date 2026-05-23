#!/usr/bin/env python3
"""
Convert HuggingFace Qwen3-ASR (Qwen3ASRForConditionalGeneration) to a single
GGUF file compatible with RapidSpeech.cpp's `Qwen3ASRModel`.

Output layout:
  general.architecture = "Qwen3ASR"
  qwen3a.encoder.*  — audio tower hparams (n_mels, n_fft, hop, d_model, etc.)
  qwen3.*           — LLM hparams (reuses the existing Qwen3 GGUF keys)
  qwen3a.mm.*       — multimodal projector dims
  tokenizer.*       — Qwen3 BPE vocab + merges
  Tensors:
    a.conv2d.{0,1,2}.{weight,bias}    — audio_tower.conv{1,2,3}
    a.conv_out.{weight,bias?}         — audio_tower.proj_out
    a.position_embd.weight            — audio_tower.embed_positions.weight
    a.blk.<N>.{attn_{q,k,v,o},attn_norm,ffn_{up,down,norm}}.{weight,bias}
    a.blk.<N>.attn_{q,k}_norm.weight  — optional
    a.post_norm.{weight,bias}         — audio_tower.layer_norm (optional)
    mm.a.mlp.{0,1}.{weight,bias}      — multi_modal_projector.linear_{1,2}
    llm.model.embed_tokens.weight, llm.model.layers.N.*, llm.model.norm.weight,
    llm.lm_head.weight                — language_model.*

Memory-layout convention follows convert_omnivoice_to_gguf.py: PyTorch weights
are kept in their native C-order buffer; only the *declared* shape is reversed
(`ravel().reshape(*reversed(shape))`) so ggml sees ne0 as the fastest axis.
This works because GGML's ne0-fastest order matches PyTorch C-order when shape
labels are reversed.

Usage:
  python scripts/convert_qwen3_asr_to_gguf.py \
      --hf-dir /path/to/Qwen3-ASR-0.6B \
      --output models/qwen3-asr-f16.gguf \
      [--f16]

Requires: transformers, safetensors, numpy, gguf.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    print("Please install gguf: pip install gguf", file=sys.stderr)
    raise


# ----------------------------------------------------------------------------
# Encoder tensor name mapping (HF -> GGUF)
# ----------------------------------------------------------------------------

def encoder_name_map(hf_name: str, n_enc_layers: int) -> str | None:
    """Return the target GGUF tensor name, or None if the HF name is not part
    of the audio tower / projector."""
    n = hf_name

    # Conv stack: audio_tower.conv{1,2,3}.{weight,bias}
    for i in (1, 2, 3):
        if n == f"audio_tower.conv{i}.weight":
            return f"a.conv2d.{i - 1}.weight"
        if n == f"audio_tower.conv{i}.bias":
            return f"a.conv2d.{i - 1}.bias"

    # proj_out: audio_tower.proj_out.{weight,bias}
    if n == "audio_tower.proj_out.weight":
        return "a.conv_out.weight"
    if n == "audio_tower.proj_out.bias":
        return "a.conv_out.bias"

    # Position embedding
    if n == "audio_tower.embed_positions.weight":
        return "a.position_embd.weight"

    # Post-encoder layer norm
    if n == "audio_tower.layer_norm.weight":
        return "a.post_norm.weight"
    if n == "audio_tower.layer_norm.bias":
        return "a.post_norm.bias"

    # Transformer layers
    prefix = "audio_tower.layers."
    if n.startswith(prefix):
        rest = n[len(prefix):]
        # split layer index off the front
        idx_str, suffix = rest.split(".", 1)
        try:
            il = int(idx_str)
        except ValueError:
            return None
        if il < 0 or il >= n_enc_layers:
            return None
        base = f"a.blk.{il}."
        mapping = {
            "self_attn.q_proj.weight":          base + "attn_q.weight",
            "self_attn.q_proj.bias":            base + "attn_q.bias",
            "self_attn.k_proj.weight":          base + "attn_k.weight",
            "self_attn.k_proj.bias":            base + "attn_k.bias",
            "self_attn.v_proj.weight":          base + "attn_v.weight",
            "self_attn.v_proj.bias":            base + "attn_v.bias",
            "self_attn.out_proj.weight":        base + "attn_o.weight",
            "self_attn.out_proj.bias":          base + "attn_o.bias",
            "self_attn.q_norm.weight":          base + "attn_q_norm.weight",
            "self_attn.q_norm.bias":            base + "attn_q_norm.bias",
            "self_attn.k_norm.weight":          base + "attn_k_norm.weight",
            "self_attn.k_norm.bias":            base + "attn_k_norm.bias",
            "self_attn_layer_norm.weight":      base + "attn_norm.weight",
            "self_attn_layer_norm.bias":        base + "attn_norm.bias",
            "fc1.weight":                       base + "ffn_up.weight",
            "fc1.bias":                         base + "ffn_up.bias",
            "fc2.weight":                       base + "ffn_down.weight",
            "fc2.bias":                         base + "ffn_down.bias",
            "final_layer_norm.weight":          base + "ffn_norm.weight",
            "final_layer_norm.bias":            base + "ffn_norm.bias",
        }
        return mapping.get(suffix)

    # Multi-modal projector
    proj_map = {
        "multi_modal_projector.linear_1.weight": "mm.a.mlp.0.weight",
        "multi_modal_projector.linear_1.bias":   "mm.a.mlp.0.bias",
        "multi_modal_projector.linear_2.weight": "mm.a.mlp.1.weight",
        "multi_modal_projector.linear_2.bias":   "mm.a.mlp.1.bias",
    }
    if n in proj_map:
        return proj_map[n]

    return None


def llm_name_map(hf_name: str) -> str | None:
    """Map `language_model.*` Qwen3 LLM tensors to RapidSpeech's `llm.*` keys."""
    if hf_name.startswith("language_model."):
        return "llm." + hf_name[len("language_model."):]
    return None


# ----------------------------------------------------------------------------
# Tensor write helper
# ----------------------------------------------------------------------------

def _store(writer: GGUFWriter, name: str, t: np.ndarray, use_f16: bool):
    """Reverse shape (PT C-order bytes -> ggml ne0-fastest labels) and emit."""
    if t.ndim >= 2:
        new_shape = tuple(reversed(t.shape))
        t = t.ravel().reshape(new_shape)
    # Pick dtype: keep 1-D / normalization tensors / biases in F32 to avoid
    # subtle numeric drift.
    eff_ndim = sum(1 for d in t.shape if d > 1)
    if use_f16 and eff_ndim >= 2 and not name.endswith(".bias") and \
            not name.endswith("norm.weight") and \
            not name.endswith("layer_norm.weight"):
        t = t.astype(np.float16)
    else:
        t = t.astype(np.float32)
    writer.add_tensor(name, t)


# ----------------------------------------------------------------------------
# Main conversion routine
# ----------------------------------------------------------------------------

def convert(hf_dir: Path, output: Path, use_f16: bool):
    print(f"Loading from {hf_dir}")
    cfg_path = hf_dir / "config.json"
    if not cfg_path.exists():
        sys.exit(f"config.json missing at {cfg_path}")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Sub-configs (Qwen3-ASR architecture wraps an audio_config + text_config).
    audio_cfg = cfg.get("audio_config", cfg.get("audio_tower_config", {}))
    text_cfg  = cfg.get("text_config", cfg.get("language_config", cfg))

    # Encoder hparams
    n_mels       = int(audio_cfg.get("num_mel_bins", 128))
    n_fft        = int(audio_cfg.get("n_fft", 400))
    hop_length   = int(audio_cfg.get("hop_length", 160))
    sample_rate  = int(audio_cfg.get("sampling_rate", 16000))
    n_window     = int(audio_cfg.get("n_window", 50))
    chunk_size   = n_window * 2
    enc_d_model  = int(audio_cfg.get("d_model", audio_cfg.get("hidden_size", 1280)))
    enc_n_head   = int(audio_cfg.get("encoder_attention_heads", audio_cfg.get("num_attention_heads", 20)))
    enc_n_layer  = int(audio_cfg.get("encoder_layers", audio_cfg.get("num_hidden_layers", 32)))
    enc_ffn_dim  = int(audio_cfg.get("encoder_ffn_dim", audio_cfg.get("intermediate_size", 5120)))
    enc_max_pos  = int(audio_cfg.get("max_source_positions", 1500))
    enc_norm_eps = float(audio_cfg.get("layer_norm_eps", 1e-5))

    # LLM hparams
    n_llm_layer  = int(text_cfg.get("num_hidden_layers", 28))
    n_llm_embd   = int(text_cfg.get("hidden_size", 1024))
    n_llm_head   = int(text_cfg.get("num_attention_heads", 16))
    n_llm_head_kv = int(text_cfg.get("num_key_value_heads", n_llm_head))
    n_llm_vocab  = int(text_cfg.get("vocab_size", 151936))
    n_llm_ff     = int(text_cfg.get("intermediate_size", 3072))
    head_dim     = int(text_cfg.get("head_dim", text_cfg.get("hidden_size", 1024) // n_llm_head))
    rope_base    = float(text_cfg.get("rope_theta", 1000000.0))
    rms_eps      = float(text_cfg.get("rms_norm_eps", 1e-6))
    n_ctx_train  = int(text_cfg.get("max_position_embeddings", 32768))
    bos          = int(text_cfg.get("bos_token_id", 151643))
    eos          = int(text_cfg.get("eos_token_id", 151645))

    # mm projector dims
    mm_in_dim  = int(cfg.get("projector_input_dim", enc_d_model))
    mm_out_dim = int(cfg.get("projector_output_dim", n_llm_embd))

    print(f"Encoder: d_model={enc_d_model} n_head={enc_n_head} n_layer={enc_n_layer} "
          f"ffn={enc_ffn_dim} n_mels={n_mels} chunk={chunk_size}")
    print(f"LLM:     n_layer={n_llm_layer} n_embd={n_llm_embd} n_head={n_llm_head} "
          f"head_dim={head_dim} vocab={n_llm_vocab}")

    writer = GGUFWriter(str(output), "Qwen3ASR")

    # Top-level metadata
    writer.add_string("general.architecture", "Qwen3ASR")
    writer.add_string("general.name", "Qwen3-ASR")

    # Encoder
    writer.add_int32("qwen3a.encoder.n_mels",          n_mels)
    writer.add_int32("qwen3a.encoder.n_fft",           n_fft)
    writer.add_int32("qwen3a.encoder.hop_length",      hop_length)
    writer.add_int32("qwen3a.encoder.sample_rate",     sample_rate)
    writer.add_int32("qwen3a.encoder.chunk_size",      chunk_size)
    writer.add_int32("qwen3a.encoder.d_model",         enc_d_model)
    writer.add_int32("qwen3a.encoder.n_head",          enc_n_head)
    writer.add_int32("qwen3a.encoder.n_layer",         enc_n_layer)
    writer.add_int32("qwen3a.encoder.ffn_dim",         enc_ffn_dim)
    writer.add_int32("qwen3a.encoder.max_source_positions", enc_max_pos)
    writer.add_float32("qwen3a.encoder.layer_norm_eps", enc_norm_eps)
    writer.add_int32("qwen3a.mm.in_dim",  mm_in_dim)
    writer.add_int32("qwen3a.mm.out_dim", mm_out_dim)

    # LLM (Qwen3) hparams — match keys consumed by llm_model::load_hparams
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
    # Reuse the FunASRNano pattern: take Qwen3's tokenizer, dump tokens +
    # merges + special token IDs into the standard `tokenizer.*` keys.
    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("Please install transformers: pip install transformers")
    tok = AutoTokenizer.from_pretrained(str(hf_dir), trust_remote_code=True)
    vocab_size = getattr(tok, "vocab_size", n_llm_vocab)

    # Build full token table (including added/special tokens) and apply the
    # GPT-2-style byte mapping so each id has a printable representation.
    byte_to_unicode = _bytes_to_unicode()
    id_to_token = {}
    base_vocab = tok.get_vocab()
    for tok_text, tid in base_vocab.items():
        id_to_token[int(tid)] = tok_text
    for tid, tok_text in tok.added_tokens_decoder.items():
        id_to_token[int(tid)] = tok_text.content if hasattr(tok_text, "content") else str(tok_text)

    max_id = max(id_to_token) if id_to_token else 0
    tokens = []
    for i in range(max(vocab_size, max_id + 1)):
        tokens.append(id_to_token.get(i, ""))

    writer.add_int32("tokenizer.vocab_size", len(tokens))
    writer.add_array("tokenizer.ggml.tokens", tokens)
    writer.add_string("tokenizer.unk_symbol", "<|endoftext|>")

    # Merges
    merges_file = hf_dir / "merges.txt"
    if merges_file.exists():
        with open(merges_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
        writer.add_array("tokenizer.ggml.merges", lines)

    # ---- Tensors ---------------------------------------------------------
    state_dict = _load_state_dict(hf_dir)
    converted_enc = 0
    converted_llm = 0
    skipped       = 0

    for name in sorted(state_dict.keys()):
        t = state_dict[name].detach().to(dtype=_t_to_dtype(state_dict[name])).cpu().numpy()
        gguf_name = encoder_name_map(name, enc_n_layer)
        if gguf_name is not None:
            _store(writer, gguf_name, t, use_f16)
            converted_enc += 1
            continue
        gguf_name = llm_name_map(name)
        if gguf_name is not None:
            _store(writer, gguf_name, t, use_f16)
            converted_llm += 1
            continue
        skipped += 1

    print(f"Encoder/projector tensors: {converted_enc}")
    print(f"LLM tensors:               {converted_llm}")
    print(f"Skipped (unmapped) tensors: {skipped}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nWritten {output}")


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _t_to_dtype(t):
    import torch
    if t.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    return torch.float32


def _load_state_dict(hf_dir: Path):
    """Load the model weights from either safetensors index or single file."""
    index = hf_dir / "model.safetensors.index.json"
    single = hf_dir / "model.safetensors"
    state = {}
    if index.exists():
        import safetensors.torch
        with open(index) as f:
            idx = json.load(f)
        files = sorted(set(idx["weight_map"].values()))
        for fn in files:
            print(f"  Loading shard {fn}")
            shard = safetensors.torch.load_file(str(hf_dir / fn), device="cpu")
            state.update(shard)
        return state
    if single.exists():
        import safetensors.torch
        print(f"  Loading {single.name}")
        return safetensors.torch.load_file(str(single), device="cpu")
    bin_index = hf_dir / "pytorch_model.bin.index.json"
    if bin_index.exists():
        import torch
        with open(bin_index) as f:
            idx = json.load(f)
        files = sorted(set(idx["weight_map"].values()))
        for fn in files:
            print(f"  Loading shard {fn}")
            shard = torch.load(str(hf_dir / fn), map_location="cpu")
            state.update(shard)
        return state
    bin_single = hf_dir / "pytorch_model.bin"
    if bin_single.exists():
        import torch
        print(f"  Loading {bin_single.name}")
        return torch.load(str(bin_single), map_location="cpu")
    sys.exit(f"No model weights found in {hf_dir}")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True, help="Local HF Qwen3-ASR checkpoint directory")
    parser.add_argument("--output", required=True, help="Output GGUF path")
    parser.add_argument("--f16", action="store_true", help="Store 2-D weights as F16 (default F32)")
    args = parser.parse_args()
    hf_dir = Path(args.hf_dir).expanduser()
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    convert(hf_dir, output, args.f16)
