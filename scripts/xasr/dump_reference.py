#!/usr/bin/env python3
"""Reference activation dumper for X-ASR zipformer2 parity testing.

Runs the fp32 ONNX (encoder/decoder/joiner) on a wav exactly as test_onnx.py
does (kaldi fbank, T=109 / shift=96 chunks, greedy transducer) and saves
intermediate tensors to an .npz the C++ port is validated against.

Saves: features, per-chunk x / encoder_out, decoder_out for each emitted hyp,
first-chunk per-token logits, final hyp + text. With --expose <v1,v2,...> it
augments the encoder graph to also output those internal value names (per chunk).
"""
import argparse, os, numpy as np
import onnx, onnxruntime as ort
import kaldi_native_fbank as knf
import soundfile as sf


def compute_feat(samples, sr):
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.samp_freq = sr
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(sr, samples.tolist())
    fb.input_finished()
    feats = np.stack([fb.get_frame(i) for i in range(fb.num_frames_ready)])
    feats = np.pad(feats, ((0, 100), (0, 0)), constant_values=0)
    return np.ascontiguousarray(feats, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--encoder", default="encoder-960ms.onnx")
    ap.add_argument("--decoder", default="decoder-960ms.onnx")
    ap.add_argument("--joiner", default="joiner-960ms.onnx")
    ap.add_argument("--tokens", default="tokens.txt")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expose", default="")
    args = ap.parse_args()
    d = args.model_dir

    enc_path = os.path.join(d, args.encoder)
    expose = [s for s in args.expose.split(",") if s]
    if expose:
        m = onnx.load(enc_path)
        existing = {o.name for o in m.graph.output}
        for v in expose:
            if v not in existing:
                m.graph.output.extend([onnx.helper.make_empty_tensor_value_info(v)])
        enc_path = os.path.join(d, "_enc_exposed.onnx")
        onnx.save(m, enc_path)

    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    enc = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(os.path.join(d, args.decoder), so, providers=["CPUExecutionProvider"])
    joi = ort.InferenceSession(os.path.join(d, args.joiner), so, providers=["CPUExecutionProvider"])

    meta = enc.get_modelmeta().custom_metadata_map
    T = int(meta["T"]); shift = int(meta["decode_chunk_len"])
    dmeta = dec.get_modelmeta().custom_metadata_map
    context_size = int(dmeta["context_size"])

    id2tok = {}
    with open(os.path.join(d, args.tokens), encoding="utf-8") as f:
        for line in f:
            t, i = line.strip().split()
            id2tok[int(i)] = t

    samples, sr = sf.read(args.wav, always_2d=True, dtype="float32")
    samples = np.ascontiguousarray(samples[:, 0])
    feats = compute_feat(samples, sr)

    enc_in = enc.get_inputs(); enc_out = enc.get_outputs()
    out_names = [o.name for o in enc_out]
    # init states
    states = []
    for n in enc_in[1:]:
        shape = [1 if isinstance(s, str) else s for s in n.shape]
        states.append(np.zeros(shape, np.int64 if "int64" in n.type else np.float32))

    blank = 0
    hyp = [blank] * context_size
    decoder_out = dec.run([dec.get_outputs()[0].name],
                          {dec.get_inputs()[0].name: np.array([hyp], np.int64)})[0]

    save = {"features": feats, "T": T, "shift": shift, "context_size": context_size}
    save["decoder_out_blank"] = decoder_out
    chunk_eouts, exposed = [], {v: [] for v in expose}
    first_logits = []
    start = ci = 0
    while start + T < feats.shape[0]:
        x = feats[start:start + T][None]
        start += shift
        feed = {enc_in[0].name: x}
        for i, n in enumerate(enc_in[1:]):
            feed[n.name] = states[i]
        out = enc.run(out_names, feed)
        named = dict(zip(out_names, out))
        encoder_out = named["encoder_out"]
        # next states = all outputs whose name starts with new_
        states = [named[n] for n in out_names if n.startswith("new_")]
        chunk_eouts.append(encoder_out)
        for v in expose:
            exposed[v].append(named[v])
        # greedy
        for k in range(encoder_out.shape[1]):
            ceo = encoder_out[0, k:k + 1]
            logit = joi.run([joi.get_outputs()[0].name],
                            {joi.get_inputs()[0].name: ceo,
                             joi.get_inputs()[1].name: decoder_out})[0]
            if ci == 0 and k < 5:
                first_logits.append(logit.copy())
            tid = int(logit.argmax())
            if tid != blank:
                hyp.append(tid)
                decoder_out = dec.run([dec.get_outputs()[0].name],
                                      {dec.get_inputs()[0].name:
                                       np.array([hyp[-context_size:]], np.int64)})[0]
        ci += 1

    save["encoder_out_chunks"] = np.stack(chunk_eouts)  # [n_chunk,1,Tc,512]
    save["first_logits"] = np.stack(first_logits) if first_logits else np.zeros(0)
    save["hyp"] = np.array(hyp[context_size:], np.int64)
    for v in expose:
        save["exposed::" + v] = np.stack(exposed[v])
    text = "".join(id2tok[i] for i in hyp[context_size:])
    save["text"] = text
    np.savez(args.out, **save)
    print(f"wrote {args.out}")
    print(f"  n_chunks={len(chunk_eouts)} encoder_out shape={chunk_eouts[0].shape}")
    print(f"  text: {text}")


if __name__ == "__main__":
    main()
