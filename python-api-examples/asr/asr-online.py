#!/usr/bin/env python3
"""
RapidSpeech Online (Streaming) ASR Example

Capture audio from the microphone (or read it from a WAV in --simulate mode),
slice it into speech segments with a neural VAD (silero-vad / firered-vad), and
transcribe each segment with the ASR model — printing transcripts as they land.

Live capture requires `sounddevice` (PortAudio). The --simulate mode replays a
WAV file as if it were streaming and works without a microphone.

Usage:
    # Live microphone — Ctrl-C to stop
    python asr-online.py \\
        --model funasr.gguf \\
        --vad silero-vad.gguf \\
        --vad-threshold 0.5

    # Replay a WAV file as a stream (no microphone needed)
    python asr-online.py \\
        --model funasr.gguf \\
        --vad silero-vad.gguf \\
        --simulate test.wav

    # 2-pass mode — CTC greedy first (fast), then LLM rescore
    python asr-online.py \\
        --model funasr.gguf \\
        --vad silero-vad.gguf \\
        --two-pass

Notes
-----
* VAD runs at 16 kHz; if the ASR model uses a different sample rate the audio
  is resampled before being pushed to ASR.
* Set --no-llm to disable the FunASRNano LLM rescoring pass for lower latency.
* Set --two-pass to emit a CTC-greedy partial as soon as the segment ends,
  then update it with the LLM-rescored result.
"""

import argparse
import math
import queue
import signal
import sys
import threading
import time
import wave

import numpy as np
import rapidspeech


SAMPLE_RATE = 16000


# ──────────────────────────────────────────────────────────────────────────────
# WAV / resampling utilities
# ──────────────────────────────────────────────────────────────────────────────
def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        i32 = (b[:, 0].astype(np.int32)
               | (b[:, 1].astype(np.int32) << 8)
               | (b[:, 2].astype(np.int32) << 16))
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
    if src_sr == dst_sr:
        return pcm
    n_out = int(math.ceil(len(pcm) * dst_sr / src_sr))
    x_src = np.arange(len(pcm), dtype=np.float64)
    x_dst = np.arange(n_out, dtype=np.float64) * (src_sr / dst_sr)
    return np.interp(x_dst, x_src, pcm).astype(np.float32)


def fmt_time(s: float) -> str:
    m = int(s // 60)
    return f"{m:02d}:{s - 60*m:05.2f}"


# ──────────────────────────────────────────────────────────────────────────────
# Streaming pipeline
# ──────────────────────────────────────────────────────────────────────────────
class StreamingASR:
    """VAD-segmented streaming ASR.

    Audio in:   `push(pcm)` accepts arbitrary chunk sizes of 16 kHz mono float32.
    Transcripts out: emitted on stdout as soon as the VAD closes a segment and
                     the segment has been transcribed.
    """

    def __init__(self, asr_ctx, vad, *, target_sr: int, use_llm: bool,
                 two_pass: bool = False):
        self.asr = asr_ctx
        self.vad = vad
        self.target_sr = target_sr
        self.use_llm = use_llm
        self.two_pass = two_pass

        # Rolling buffer keeps up to MAX_BUFFER_S of audio so we can slice
        # [start_s, end_s] out of it when the VAD reports a closed segment.
        self.MAX_BUFFER_S = 60.0
        self.audio_buf: list[np.ndarray] = []
        self.buf_origin_s = 0.0  # absolute timestamp of audio_buf[0][0]
        self.total_pushed_s = 0.0

    def push(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        self.audio_buf.append(pcm.astype(np.float32, copy=False))
        self.total_pushed_s += pcm.size / SAMPLE_RATE
        self.vad.push_audio(pcm)

        # Bound rolling buffer.
        total = sum(len(b) for b in self.audio_buf)
        max_samples = int(self.MAX_BUFFER_S * SAMPLE_RATE)
        while total > max_samples and len(self.audio_buf) > 1:
            dropped = len(self.audio_buf.pop(0))
            total -= dropped
            self.buf_origin_s += dropped / SAMPLE_RATE

        # Drain any closed segments and transcribe.
        for seg in self.vad.drain_segments():
            self._transcribe_segment(seg["start_s"], seg["end_s"])

    def flush(self) -> None:
        """Force an end-of-stream — any audio currently considered speech is
        flushed as a final segment."""
        if not self.vad.is_speech():
            return
        end_s = self.total_pushed_s
        # The last segment_start is unknown to us — approximate with the last
        # speech_start frame.
        frames = self.vad.drain_frames(8192)
        start_s = None
        for f in reversed(frames):
            if f.get("is_speech_start"):
                # frame_idx is 1-based; convert to seconds.
                start_s = (f["frame_idx"] - 1) * (1.0 / self._vad_fps())
                break
        if start_s is None:
            start_s = max(0.0, end_s - 1.0)
        if end_s - start_s > 0.2:
            self._transcribe_segment(start_s, end_s)

    def _vad_fps(self) -> float:
        # silero-vad: 16000 / 512 ≈ 31.25 ; firered-vad: 100 fps
        return 100.0 if self.vad.get_arch() == "firered-vad" else 16000.0 / 512.0

    def _slice(self, start_s: float, end_s: float) -> np.ndarray | None:
        """Pull [start_s, end_s] out of the rolling 16k buffer."""
        s0 = int(max(0.0, (start_s - self.buf_origin_s) * SAMPLE_RATE))
        s1 = int(max(0.0, (end_s   - self.buf_origin_s) * SAMPLE_RATE))
        if s1 <= s0:
            return None
        out = np.empty(s1 - s0, dtype=np.float32)
        off = 0
        src_off = 0
        for b in self.audio_buf:
            nxt = src_off + len(b)
            if nxt <= s0:
                src_off = nxt
                continue
            if src_off >= s1:
                break
            frm = max(0, s0 - src_off)
            to  = min(len(b), s1 - src_off)
            chunk = b[frm:to]
            out[off:off + len(chunk)] = chunk
            off += len(chunk)
            src_off = nxt
        return out[:off]

    def _transcribe_segment(self, start_s: float, end_s: float) -> None:
        clip16k = self._slice(start_s, end_s)
        if clip16k is None or clip16k.size < int(0.2 * SAMPLE_RATE):
            return
        # Resample to the ASR model's native rate if needed.
        if self.target_sr != SAMPLE_RATE:
            clip = resample_linear(clip16k, SAMPLE_RATE, self.target_sr)
        else:
            clip = clip16k

        dur = (end_s - start_s)
        tag = f"[{fmt_time(start_s)} → {fmt_time(end_s)} | {dur:4.2f}s"

        self.asr.reset()
        self.asr.push_audio(np.ascontiguousarray(clip))

        if self.two_pass:
            # Pass 1: CTC greedy (fast)
            self.asr.set_use_llm(False)
            t0 = time.perf_counter()
            self.asr.process()
            t1 = time.perf_counter()
            ctc_text = self.asr.get_text() or "(no speech)"
            rtf1 = (t1 - t0) / max(dur, 1e-9)
            print(f"{tag} | CTC {rtf1:4.2f}] {ctc_text}", flush=True)

            # Pass 2: LLM rescore
            self.asr.set_use_llm(True)
            t0 = time.perf_counter()
            self.asr.redecode()
            t1 = time.perf_counter()
            llm_text = self.asr.get_text() or "(no speech)"
            rtf2 = (t1 - t0) / max(dur, 1e-9)
            pad = " " * (len(tag) - 3)
            print(f"{pad}| LLM {rtf2:4.2f}] {llm_text}", flush=True)
        else:
            if not self.use_llm:
                self.asr.set_use_llm(False)
            t0 = time.perf_counter()
            self.asr.process()
            elapsed = time.perf_counter() - t0
            text = self.asr.get_text() or "(no speech)"
            rtf = elapsed / max(dur, 1e-9)
            print(f"{tag} | RTF {rtf:4.2f}] {text}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Live capture and simulation
# ──────────────────────────────────────────────────────────────────────────────
def run_live(pipeline: StreamingASR, *, device: int | None, block_ms: int) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(
            "sounddevice is required for live capture. Install it with:\n"
            "    pip install sounddevice\n"
            "Or run with --simulate <wav> to replay a file instead."
        )

    blocksize = max(1, int(SAMPLE_RATE * block_ms / 1000))
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    stop_evt = threading.Event()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"[audio] status: {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    def handle_sigint(_sig, _frm):
        stop_evt.set()

    signal.signal(signal.SIGINT, handle_sigint)

    print(f"Listening (Ctrl-C to stop) — device={device}, block={block_ms}ms, sr={SAMPLE_RATE}Hz")
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=blocksize,
                        device=device,
                        callback=callback):
        while not stop_evt.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            pipeline.push(chunk)

    print("\nStopping — flushing remaining audio...")
    pipeline.flush()


def run_simulate(pipeline: StreamingASR, wav_path: str, *,
                 block_ms: int, realtime: bool) -> None:
    pcm, sr = read_wav(wav_path)
    if sr != SAMPLE_RATE:
        print(f"Resampling {sr} → {SAMPLE_RATE} Hz for streaming")
        pcm = resample_linear(pcm, sr, SAMPLE_RATE)
    print(f"Simulating stream from {wav_path}: {len(pcm)/SAMPLE_RATE:.2f}s "
          f"(realtime={realtime})")

    block = max(1, int(SAMPLE_RATE * block_ms / 1000))
    block_period = block_ms / 1000.0
    t_next = time.perf_counter()
    for i in range(0, len(pcm), block):
        chunk = pcm[i:i + block]
        pipeline.push(chunk)
        if realtime:
            t_next += block_period
            sleep_for = t_next - time.perf_counter()
            if sleep_for > 0:
                time.sleep(sleep_for)

    pipeline.flush()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="RapidSpeech Online (streaming) ASR")
    p.add_argument("--model", required=True, help="Path to GGUF ASR model")
    p.add_argument("--vad", required=True, help="Path to silero-vad / firered-vad GGUF")
    p.add_argument("--threads", type=int, default=4, help="ASR CPU threads (default: 4)")
    p.add_argument("--gpu", type=int, default=1, help="ASR on GPU? 0/1 (default: 1)")
    p.add_argument("--vad-threshold", type=float, default=0.5,
                   help="VAD speech threshold (default: 0.5)")
    p.add_argument("--vad-gpu", type=int, default=0,
                   help="VAD on GPU? 0/1 (default: 0 — tiny models, CPU is faster)")
    p.add_argument("--no-llm", action="store_true",
                   help="Disable FunASRNano LLM rescoring (lower latency, CTC-only)")
    p.add_argument("--two-pass", action="store_true",
                   help="Emit CTC-greedy text first, then LLM-rescored text (FunASRNano)")
    p.add_argument("--prompt", default=None, help="LLM decoder prompt")
    p.add_argument("--block-ms", type=int, default=64,
                   help="Stream block size in ms (default: 64)")
    p.add_argument("--device", type=int, default=None,
                   help="Microphone device index (default: system default)")
    p.add_argument("--simulate", default=None,
                   help="Replay a WAV file as a stream instead of using the microphone")
    p.add_argument("--simulate-realtime", action="store_true",
                   help="With --simulate, throttle playback to wall-clock rate")
    p.add_argument("--list-devices", action="store_true",
                   help="List available audio input devices and exit")
    args = p.parse_args()

    if args.two_pass and args.no_llm:
        p.error("--two-pass and --no-llm are mutually exclusive")

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    print(f"rapidspeech version: {rapidspeech.version()}")
    print(f"Loading ASR: {args.model}")
    asr = rapidspeech.asr_offline(
        model_path=args.model,
        n_threads=args.threads,
        use_gpu=bool(args.gpu),
    )
    meta = asr.get_model_meta()
    target_sr = meta["audio_sample_rate"]
    print(f"  arch={meta['arch_name']}  sr={target_sr}  backend={asr.get_backend_name()}")

    if args.prompt:
        asr.set_user_input_prompt(args.prompt)
    if args.no_llm:
        asr.set_use_llm(False)

    print(f"Loading VAD: {args.vad}")
    vad = rapidspeech.vad(model_path=args.vad,
                          n_threads=max(2, args.threads // 2),
                          use_gpu=bool(args.vad_gpu))
    vad.set_threshold(args.vad_threshold)
    print(f"  arch={vad.get_arch()}  threshold={args.vad_threshold}\n")

    pipeline = StreamingASR(asr, vad, target_sr=target_sr,
                            use_llm=not args.no_llm,
                            two_pass=args.two_pass)

    if args.simulate:
        run_simulate(pipeline, args.simulate,
                     block_ms=args.block_ms,
                     realtime=args.simulate_realtime)
    else:
        run_live(pipeline, device=args.device, block_ms=args.block_ms)

    print("Done.")


if __name__ == "__main__":
    main()
