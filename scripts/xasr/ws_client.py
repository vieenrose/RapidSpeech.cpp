import asyncio, json, sys, time, numpy as np, soundfile as sf, websockets
async def main(url, wav):
    data,sr = sf.read(wav, dtype="int16", always_2d=True); pcm=data[:,0]
    if sr!=16000: print("need 16k"); return
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type":"start","sample_rate":16000}))
        print("<", await ws.recv())
        chunk=1600  # 100ms
        async def reader():
            async for msg in ws:
                m=json.loads(msg)
                if m["type"]=="partial": print("partial:", m["text"])
                elif m["type"]=="final": print("FINAL:", m["text"], "first_partial_latency=", m.get("first_partial_latency")); return
        rt=asyncio.create_task(reader())
        for i in range(0,len(pcm),chunk):
            await ws.send(pcm[i:i+chunk].tobytes()); await asyncio.sleep(0.0)
        await ws.send(json.dumps({"type":"end"}))
        await rt
asyncio.run(main(sys.argv[1], sys.argv[2]))
