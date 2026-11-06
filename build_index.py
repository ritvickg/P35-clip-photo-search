"""Run once after putting 200+ photos in photos/ :  python build_index.py"""
from core import build_index
print("Indexed", build_index(), "photos")
