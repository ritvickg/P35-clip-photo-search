"""Downloads 240 ready-made photos + their captions (Flickr, educational use) into photos/ and assets/captions.json."""
import json, random
from pathlib import Path
from datasets import load_dataset

ds = None
for name, split in [("jxie/flickr8k", "train"), ("nlphuji/flickr30k", "test")]:
    try:
        ds = load_dataset(name, split=split); print("using dataset", name); break
    except Exception as e:
        print("could not load", name, "->", str(e)[:150])
assert ds is not None, "No dataset could be downloaded - send the messages above to Claude."

idx = list(range(len(ds))); random.Random(0).shuffle(idx)
Path("photos").mkdir(exist_ok=True); Path("assets").mkdir(exist_ok=True)
caps = {}
for n, i in enumerate(idx[:240]):
    row = ds[i]; name = f"img_{n:04d}.jpg"
    im = row["image"].convert("RGB"); im.thumbnail((512, 512)); im.save(f"photos/{name}", quality=85)
    c = []
    for k, v in row.items():
        if "caption" in k:
            c += [v] if isinstance(v, str) else [x for x in v if isinstance(x, str)]
    caps[name] = c
json.dump(caps, open("assets/captions.json", "w"))
print("saved", len(caps), "photos; example:", list(caps.items())[0])
