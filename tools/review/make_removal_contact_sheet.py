#!/usr/bin/env python3
"""Create a compact before/after QA sheet for Phase 3."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

original = Image.open("samples/input.jpg").convert("RGB")
cleaned = Image.open("output/cleaned.png").convert("RGB")
groups = json.loads(Path("output/translations.json").read_text())["groups"]
margin = 35
cell_width = 650
rows = []
for group in groups:
    box = group["bbox"]
    left = max(0, box["x"] - margin)
    top = max(0, box["y"] - margin)
    right = min(original.width, box["x"] + box["width"] + margin)
    bottom = min(original.height, box["y"] + box["height"] + margin)
    before = original.crop((left, top, right, bottom))
    after = cleaned.crop((left, top, right, bottom))
    scale = min(cell_width / max(1, before.width), 2.0)
    size = (round(before.width * scale), round(before.height * scale))
    rows.append((group["id"], before.resize(size), after.resize(size)))

row_heights = [max(a.height, b.height) + 40 for _, a, b in rows]
sheet = Image.new("RGB", (cell_width * 2 + 30, sum(row_heights)), "#202020")
draw = ImageDraw.Draw(sheet)
y = 0
for (label, before, after), row_height in zip(rows, row_heights):
    draw.text((5, y + 5), label + "  BEFORE", fill="white")
    draw.text((cell_width + 20, y + 5), "AFTER", fill="white")
    sheet.paste(before, (5, y + 25))
    sheet.paste(after, (cell_width + 20, y + 25))
    y += row_height
sheet.save("output/removal-contact-sheet.png", optimize=True)

