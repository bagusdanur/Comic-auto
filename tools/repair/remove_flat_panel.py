#!/usr/bin/env python3
"""Remove lettering from an existing flat panel without changing its geometry."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--box", required=True, help="left,top,right,bottom")
args = parser.parse_args()
box = tuple(int(value) for value in args.box.split(","))
if len(box) != 4:
    raise SystemExit("box needs four integers")
image = np.asarray(Image.open(args.source).convert("RGB")).copy()
x1, y1, x2, y2 = box
sample = np.concatenate((image[y1:y1 + 8, x1:x2].reshape(-1, 3),
                         image[y1:y2, x1:x1 + 8].reshape(-1, 3)), axis=0)
color = np.median(sample, axis=0).round().astype(np.uint8)
image[y1:y2, x1:x2] = color
Image.fromarray(image).save(args.output, format="PNG", optimize=True)
print(f"panel={box} color={color.tolist()}")

