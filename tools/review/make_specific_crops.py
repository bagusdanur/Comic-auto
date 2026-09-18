#!/usr/bin/env python3
from PIL import Image, ImageDraw
import sys

specs = [
    (sys.argv[1], (300, 430, 720, 850), "page 4 - panel narration"),
    (sys.argv[1], (80, 8050, 540, 8700), "page 4 - oval shout"),
    (sys.argv[2], (30, 11920, 520, 12600), "page 14 - oval dialogue"),
]
rows = []
for path, box, label in specs:
    crop = Image.open(path).convert("RGB").crop(box)
    canvas = Image.new("RGB", (crop.width, crop.height + 30), "#202020")
    ImageDraw.Draw(canvas).text((6, 7), label, fill="white")
    canvas.paste(crop, (0, 30))
    rows.append(canvas)
sheet = Image.new("RGB", (max(x.width for x in rows), sum(x.height for x in rows)), "#111111")
y = 0
for row in rows:
    sheet.paste(row, (0, y)); y += row.height
sheet.save(sys.argv[3], quality=92)

