#!/usr/bin/env python3
"""Replace one missed flat-background caption while preserving its footprint."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("image", type=Path)
parser.add_argument("--box", required=True, help="left,top,right,bottom")
parser.add_argument("--text", required=True)
parser.add_argument("--font", type=Path, required=True)
parser.add_argument("--fill", default="245,28,18")
args = parser.parse_args()

box = tuple(int(value) for value in args.box.split(","))
color = tuple(int(value) for value in args.fill.split(","))
image = Image.open(args.image).convert("RGB")
array = np.asarray(image).copy()
x1, y1, x2, y2 = box
band = 8
border = np.concatenate((
    array[max(0, y1-band):y1, x1:x2].reshape(-1, 3),
    array[y2:min(array.shape[0], y2+band), x1:x2].reshape(-1, 3),
    array[y1:y2, max(0, x1-band):x1].reshape(-1, 3),
    array[y1:y2, x2:min(array.shape[1], x2+band)].reshape(-1, 3),
))
background = tuple(np.median(border, axis=0).round().astype(np.uint8))
array[y1:y2, x1:x2] = background
image = Image.fromarray(array)
draw = ImageDraw.Draw(image)
text = args.text.replace("\\n", "\n")
for size in range(min(150, y2-y1), 15, -1):
    font = ImageFont.truetype(str(args.font), size)
    bounds = draw.multiline_textbbox((0, 0), text, font=font, spacing=round(size*0.12), align="center")
    width, height = bounds[2]-bounds[0], bounds[3]-bounds[1]
    if width <= (x2-x1)*0.90 and height <= (y2-y1)*0.86:
        break
draw.multiline_text(((x1+x2)//2, (y1+y2)//2), text, font=font, fill=color,
                    spacing=round(size*0.12), align="center", anchor="mm",
                    stroke_width=max(1, round(size*0.025)), stroke_fill=(20, 0, 0))
image.save(args.image, format="PNG", optimize=True)

