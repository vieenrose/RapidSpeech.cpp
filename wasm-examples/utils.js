/**
 * Shared helpers used by the per-tab demo pages.
 * Exposed under window.RSUtils.
 */
(function () {
  const TARGET_SR = 16000;

  // Read a File, decode any browser-supported audio container, mix to mono,
  // resample to 16 kHz via OfflineAudioContext. Returns Float32Array.
  async function decodeFileToMono16k(file) {
    return decodeFileToMonoAt(file, TARGET_SR);
  }

  // Same as above but with caller-chosen target sample rate (used by TTS
  // voice-cloning, which wants the model's native rate).
  async function decodeFileToMonoAt(file, targetSr) {
    const buf = await file.arrayBuffer();
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    let decoded;
    try {
      decoded = await ctx.decodeAudioData(buf);
    } finally {
      ctx.close().catch(() => {});
    }
    // Downmix to mono by averaging channels.
    const n = decoded.length;
    const mono = new Float32Array(n);
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const src = decoded.getChannelData(ch);
      for (let i = 0; i < n; i++) mono[i] += src[i];
    }
    if (decoded.numberOfChannels > 1) {
      const inv = 1 / decoded.numberOfChannels;
      for (let i = 0; i < n; i++) mono[i] *= inv;
    }
    if (decoded.sampleRate === targetSr) return mono;
    return resample(mono, decoded.sampleRate, targetSr);
  }

  // Resample float32 PCM from `fromRate` to `toRate` via OfflineAudioContext.
  async function resample(pcm, fromRate, toRate) {
    if (fromRate === toRate) return pcm;
    const targetLen = Math.ceil(pcm.length * toRate / fromRate);
    const off = new OfflineAudioContext(1, targetLen, toRate);
    const buf = off.createBuffer(1, pcm.length, fromRate);
    buf.copyToChannel(pcm, 0);
    const src = off.createBufferSource();
    src.buffer = buf;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    return rendered.getChannelData(0);
  }

  // Encode Float32 mono PCM into a 16-bit PCM WAV Blob.
  function encodeWav(samples, sampleRate) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buf);
    const writeStr = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);              // PCM
    view.setUint16(22, 1, true);              // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); // byte rate
    view.setUint16(32, 2, true);              // block align
    view.setUint16(34, 16, true);             // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, samples.length * 2, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = (seconds % 60).toFixed(1);
    return String(m).padStart(2, '0') + ':' + String(s).padStart(4, '0');
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
  }

  // setStatus helper bound to a given <span> element.
  function statusSetter(el) {
    return (msg, cls = 'idle') => {
      el.textContent = msg;
      el.className = 'status ' + cls;
    };
  }

  window.RSUtils = {
    TARGET_SR,
    decodeFileToMono16k,
    decodeFileToMonoAt,
    resample,
    encodeWav,
    formatTime,
    escapeHtml,
    statusSetter,
  };
})();
