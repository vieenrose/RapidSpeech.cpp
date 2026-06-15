# Extract sherpa's gold token-id sequence (concatenated over sentences) for each test line.
import sherpa_onnx, sys, io, contextlib, re, os
b="/tmp/m8k"
cfg=sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
  matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(acoustic_model=f"{b}/model-steps-3.onnx",
    vocoder=f"{b}/vocos-8khz-univ.onnx", lexicon=f"{b}/lexicon.txt", tokens=f"{b}/tokens.txt",
    data_dir="/tmp/matcha-icefall-zh-en/espeak-ng-data"), debug=True, num_threads=1), max_num_sentences=1)
tts=sherpa_onnx.OfflineTts(cfg)
import ctypes
# sherpa logs go to stderr via SHERPA_ONNX_LOGE; capture fd 2
def gen(text):
    r,w=os.pipe()
    se=os.dup(2); os.dup2(w,2)
    try: tts.generate(text, sid=0, speed=1.0)
    finally: os.dup2(se,2); os.close(w)
    out=os.read(r,1<<20).decode('utf8','replace'); os.close(r)
    ids=[]
    for m in re.finditer(r'new sentence: \[([0-9, ]*)\]', out):
        ids += [int(x) for x in m.group(1).split(',') if x.strip()]
    return ids
for line in sys.argv[1:]:
    print("GOLD\t"+line+"\t"+" ".join(map(str,gen(line))))
