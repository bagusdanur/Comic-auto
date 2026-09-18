#!/usr/bin/env python3
"""Build an A/B crop sheet around translated group boxes."""

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw


left = Image.open(sys.argv[1]).convert("RGB")
right = Image.open(sys.argv[2]).convert("RGB")
groups = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8")).get("groups", [])
rows = []
for group in groups:
    box = group["bbox"]
    pad = 55
    x1 = max(0, int(box["x"]) - pad)
    y1 = max(0, int(box["y"]) - pad)
    x2 = min(left.width, int(box["x"] + box["width"]) + pad)
    y2 = min(left.height, int(box["y"] + box["height"]) + pad)
    a = left.crop((x1, y1, x2, y2))
    b = right.crop((x1, y1, x2, y2))
    width = max(a.width, b.width)
    row = Image.new("RGB", (width * 2 + 12, max(a.height, b.height) + 28), "#222222")
    row.paste(a, (0, 28))
    row.paste(b, (width + 12, 28))
    draw = ImageDraw.Draw(row)
    draw.text((5, 5), f"{group['id']} HYBRID", fill="white")
    draw.text((width + 17, 5), "FULL XIANSCAN", fill="white")
    rows.append(row)
sheet = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "#111111")
y = 0
for row in rows:
    sheet.paste(row, (0, y))
    y += row.height
sheet.save(sys.argv[4], quality=90)

