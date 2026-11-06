"""Resize your photos to max 512px so the repo/Space stays small.  python prepare_photos.py <source_folder>"""
import sys
from pathlib import Path
from PIL import Image
src, dst = Path(sys.argv[1]), Path(__file__).parent / "photos"
dst.mkdir(exist_ok=True)
n = 0
for f in src.rglob("*"):
    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        im = Image.open(f).convert("RGB"); im.thumbnail((512, 512))
        im.save(dst / f"img_{n:04d}.jpg", quality=85); n += 1
print("Saved", n, "photos to", dst)
