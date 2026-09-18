#!/usr/bin/env python3
"""Resize and paste a restored crop into an exact region of a page."""

import argparse
from pathlib import Path

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("page", type=Path)
parser.add_argument("restoration", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--box", required=True, help="left,top,right,bottom")
args = parser.parse_args()
box = tuple(int(value) for value in args.box.split(","))
if len(box) != 4:
    raise SystemExit("box needs four integers")
left, top, right, bottom = box
with Image.open(args.page) as opened:
    page = opened.convert("RGB")
with Image.open(args.restoration) as opened:
    restored = opened.convert("RGB").resize((right - left, bottom - top), Image.Resampling.LANCZOS)
page.paste(restored, (left, top))
page.save(args.output, format="PNG", optimize=True)

