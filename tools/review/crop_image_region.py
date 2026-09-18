#!/usr/bin/env python3
"""Crop an exact image region for visual QA or targeted restoration."""

import argparse
from pathlib import Path

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--box", required=True, help="left,top,right,bottom")
args = parser.parse_args()
box = tuple(int(value) for value in args.box.split(","))
if len(box) != 4:
    raise SystemExit("box needs four integers")
with Image.open(args.source) as image:
    crop = image.crop(box)
    if args.output.suffix.casefold() in {".jpg", ".jpeg"}:
        crop.convert("RGB").save(args.output, format="JPEG", quality=95, optimize=True)
    else:
        crop.save(args.output, format="PNG", optimize=True)

