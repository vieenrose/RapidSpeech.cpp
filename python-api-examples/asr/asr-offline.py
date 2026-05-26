#!/usr/bin/env python3
"""
RapidSpeech Offline ASR Example

Transcribe a WAV file using a GGUF ASR model. Supports:
  - Any WAV bit-depth (8/16/24/32-bit PCM) and any sample rate (auto-resampled)
  - Multi-channel input (averaged to mono)
  - 2-pass decoding: CTC-greedy (fast) followed by LLM rescoring (accurate)
  - Optional CTC pre-check to skip LLM on silence
  - Optional VAD pre-segmentation (silero-vad or firered-vad) — the clip is
    split into speech regions and each region is transcribed separately

Usage:
    # Basic
    python asr-offline.py --model funasr.gguf --audio test.wav

    # 2-pass (CTC + LLM rescore)
    python asr-offline.py --model funasr.gguf --audio test.wav --two-pass

    # With VAD segmentation (silero-vad / firered-vad auto-detected from GGUF)
    python asr-offline.py --model funasr.gguf --audio long.wav \\
        --vad silero-vad.gguf --vad-threshold 0.5 --vad-min-seg 0.3

    # Benchmark
    python asr-offline.py --model funasr.gguf --audio test.wav --runs 5
"""

import argparse
import math
import time
import wave

import numpy as np
import rapidspeech


# ──────────────────────────────────────────────────────────────────────────────
# WAV reader with resampling
# ──────────────────────────────────────────────────────────────────────────────
def read_wav(file_path: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (float32 mono PCM in [-1,1], sample_rate)."""
    with wave.open(file_path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        nf = wf.getnframes()
        raw = wf.readframes(nf)

    if sw == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        # 24-bit little-endian packed; expand to int32 then normalise
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
        # Sign-extend 24-bit → 32-bit
        i32 = np.where(i32 & 0x800000, i32 | ~0xFFFFFF, i32)
        pcm = i32.astype(np.float32) / (1 << 23)
    elif sw == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / (1 << 31)
    else:
        raise ValueError(f"Unsupported sample width: {sw}")

    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)

    return np.ascontiguousarray(pcm, dtype=np.float32), sr


def resample_linear(pcm: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear-interpolation resample. Good enough as a Python fallback."""
    if src_sr == dst_sr:
        return pcm
    n_out = int(math.ceil(len(pcm) * dst_sr / src_sr))
    x_src = np.arange(len(pcm), dtype=np.float64)
    x_dst = np.arange(n_out, dtype=np.float64) * (src_sr / dst_sr)
    return np.interp(x_dst, x_src, pcm).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Transcription helpers
# ──────────────────────────────────────────────────────────────────────────────
def transcribe_segment(ctx, pcm: np.ndarray, *, two_pass: bool) -> tuple[str, str | None, float, float]:
    """Run a single (CTC or CTC+LLM) inference on `pcm`. Returns
    (first_pass_text, second_pass_text_or_None, t_first, t_second_or_0)."""
    ctx.reset()
    ctx.push_audio(pcm)

    if two_pass:
        ctx.set_use_llm(False)
    t0 = time.perf_counter()
    ctx.process()
    t1 = time.perf_counter()
    first = ctx.get_text()
    t_first = t1 - t0

    if not two_pass:
        return first, None, t_first, 0.0

    ctx.set_use_llm(True)
    t0 = time.perf_counter()
    ctx.redecode()
    t1 = time.perf_counter()
    return first, ctx.get_text(), t_first, t1 - t0


def fmt_time(s: float) -> str:
    m = int(s // 60)
    return f"{m:02d}:{s - 60*m:05.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="RapidSpeech Offline ASR")
    parser.add_argument("--model", required=True, help="Path to GGUF ASR model")
    parser.add_argument("--audio", required=True, help="Path to WAV file")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads (default: 4)")
    parser.add_argument("--gpu", type=int, default=1, help="Use GPU: 1=on, 0=off (default: 1)")
    parser.add_argument("--runs", type=int, default=1, help="Inference runs for benchmarking (default: 1)")
    parser.add_argument("--prompt", type=str, default=None, help="LLM decoder prompt")
    parser.add_argument("--two-pass", action="store_true",
                        help="Run CTC-greedy first, then LLM rescore (FunASRNano)")
    parser.add_argument("--ctc-precheck", action="store_true",
                        help="Skip LLM on silence by running a quick CTC precheck first")
    # VAD
    parser.add_argument("--vad", type=str, default=None,
                        help="Path to a silero-vad / firered-vad GGUF for pre-segmentation")
    parser.add_argument("--vad-threshold", type=float, default=0.5,
                        help="VAD speech threshold (default: 0.5)")
    parser.add_argument("--vad-min-seg", type=float, default=0.3,
                        help="Drop segments shorter than this (seconds, default: 0.3)")
    parser.add_argument("--vad-gpu", type=int, default=0,
                        help="Run VAD on GPU? 0/1 (default: 0 — tiny models, CPU is faster)")
    args = parser.parse_args()

    print(f"rapidspeech version: {rapidspeech.version()}")
    print(f"Loading ASR model:  {args.model}")
    ctx = rapidspeech.asr_offline(
        model_path=args.model,
        n_threads=args.threads,
        use_gpu=bool(args.gpu),
    )

    meta = ctx.get_model_meta()
    target_sr = meta["audio_sample_rate"]
    print(f"  arch={meta['arch_name']}  sr={target_sr} Hz  backend={ctx.get_backend_name()}")

    if args.prompt:
        ctx.set_user_input_prompt(args.prompt)
        print(f"  prompt: {args.prompt}")
    if args.ctc_precheck:
        ctx.set_ctc_precheck(True)
        print("  CTC pre-check: ON")

    vad = None
    if args.vad:
        print(f"Loading VAD model:  {args.vad}")
        vad = rapidspeech.vad(model_path=args.vad,
                              n_threads=max(2, args.threads // 2),
                              use_gpu=bool(args.vad_gpu))
        vad.set_threshold(args.vad_threshold)
        print(f"  arch={vad.get_arch()}  threshold={args.vad_threshold}")

    # Read & resample audio to the model's native rate
    pcm, sr = read_wav(args.audio)
    if sr != target_sr:
        print(f"Resampling {sr} → {target_sr} Hz")
        pcm = resample_linear(pcm, sr, target_sr)
    duration = len(pcm) / target_sr
    print(f"Audio: {args.audio}  ({len(pcm)} samples, {duration:.2f}s)\n")

    # ── VAD-driven path: segment, transcribe each region ───────────
    if vad is not None:
        # VAD always runs at 16 kHz mono — if the ASR sr differs we make a
        # parallel 16k copy for VAD only.
        if target_sr == 16000:
            vad_pcm = pcm
        else:
            vad_pcm = resample_linear(pcm, target_sr, 16000)
        segments = vad.detect_full(vad_pcm)
        segments = [s for s in segments if s["end_s"] - s["start_s"] >= args.vad_min_seg]
        print(f"VAD: {len(segments)} segment(s) ≥ {args.vad_min_seg}s")
        if not segments:
            print("  (no speech detected by VAD)")
            return

        total_t = 0.0
        for i, seg in enumerate(segments):
            s0 = int(max(0.0, seg["start_s"]) * target_sr)
            s1 = int(min(duration, seg["end_s"]) * target_sr)
            if s1 <= s0:
                continue
            slice_ = np.ascontiguousarray(pcm[s0:s1])
            first, second, t1, t2 = transcribe_segment(ctx, slice_, two_pass=args.two_pass)
            total_t += t1 + t2
            seg_dur = (s1 - s0) / target_sr
            tag = f"[{fmt_time(seg['start_s'])} → {fmt_time(seg['end_s'])} | {seg_dur:.2f}s]"
            if args.two_pass:
                print(f"{tag} CTC: {first or '(no speech)'}")
                print(f"{' ' * len(tag)} LLM: {second or '(no speech)'}")
            else:
                print(f"{tag} {first or '(no speech)'}")
        print(f"\nTotal inference: {total_t:.2f}s, audio: {duration:.2f}s, RTF={total_t/duration:.3f}")
        return

    # ── Whole-clip path (no VAD) ───────────────────────────────────
    elapsed_first = []
    elapsed_second = []
    for i in range(args.runs):
        first, second, t_first, t_second = transcribe_segment(
            ctx, pcm, two_pass=args.two_pass)
        elapsed_first.append(t_first)
        if args.two_pass:
            elapsed_second.append(t_second)

        if args.runs == 1:
            if args.two_pass:
                print(f"  CTC :  {first or '(no speech)'}")
                print(f"  LLM :  {second or '(no speech)'}")
            else:
                print(f"  Result: {first or '(no speech)'}")
        else:
            rtf = t_first / duration
            print(f"  Run {i+1}: {t_first:.3f}s  RTF={rtf:.3f}  "
                  f"{(second or first) or '(no speech)'}")

    if args.runs > 1:
        avg1 = sum(elapsed_first) / len(elapsed_first)
        print(f"\n  CTC avg: {avg1:.3f}s  RTF={avg1/duration:.3f}")
        if elapsed_second:
            avg2 = sum(elapsed_second) / len(elapsed_second)
            print(f"  LLM avg: {avg2:.3f}s  RTF={avg2/duration:.3f}")


if __name__ == "__main__":
    main()
