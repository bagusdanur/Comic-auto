#!/usr/bin/env python3
"""Create a compact clean/typeset QA sheet for Phase 4."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("--cleaned", type=Path, default=Path("output/cleaned.png"))
parser.add_argument("--typeset", type=Path, default=Path("output/typeset.png"))
parser.add_argument("--translations", type=Path, default=Path("output/translations.json"))
parser.add_argument("--output", type=Path, default=Path("output/typeset-contact-sheet.png"))
args = parser.parse_args()
cleaned = Image.open(args.cleaned).convert("RGB")
typeset = Image.open(args.typeset).convert("RGB")
groups = json.loads(args.translations.read_text())["groups"]
margin, cell_width = 40, 650
rows = []
for group in groups:
    box = group["bbox"]
    area = (
        max(0, box["x"] - margin), max(0, box["y"] - margin),
        min(cleaned.width, box["x"] + box["width"] + margin),
        min(cleaned.height, box["y"] + box["height"] + margin),
    )
    before, after = cleaned.crop(area), typeset.crop(area)
    scale = min(cell_width / max(1, before.width), 2.0)
    size = (round(before.width * scale), round(before.height * scale))
    rows.append((group["id"], before.resize(size), after.resize(size)))

heights = [max(a.height, b.height) + 40 for _, a, b in rows]
sheet = Image.new("RGB", (cell_width * 2 + 30, sum(heights)), "#202020")
draw = ImageDraw.Draw(sheet)
y = 0
for (label, before, after), height in zip(rows, heights):
    draw.text((5, y + 5), label + "  CLEAN", fill="white")
    draw.text((cell_width + 20, y + 5), "TYPESET", fill="white")
    sheet.paste(before, (5, y + 25))
    sheet.paste(after, (cell_width + 20, y + 25))
    y += height
sheet.save(args.output, optimize=True)

