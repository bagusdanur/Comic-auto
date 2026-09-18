#!/usr/bin/env python3
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

root = Path(sys.argv[1])
items = []
for report_path in sorted((root / "metadata").glob("page-*/branding-report.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("replacements"):
        continue
    page_no = report_path.parent.name.split("-")[-1]
    original = next((root / "original").glob(f"{page_no}_*"))
    output = next((root / "output").glob(f"{page_no}_*.png"))
    with Image.open(original) as src, Image.open(output) as dst:
        src, dst = src.convert("RGB"), dst.convert("RGB")
        for index, replacement in enumerate(report["replacements"], 1):
            x1, y1, x2, y2 = replacement["covered"]
            pad_x, pad_y = 100, 90
            crop_box = (max(0, x1-pad_x), max(0, y1-pad_y), min(src.width, x2+pad_x), min(src.height, y2+pad_y))
            a, b = src.crop(crop_box), dst.crop(crop_box)
            scale = min(1.0, 420 / max(a.width, 1))
            size = (max(1, round(a.width*scale)), max(1, round(a.height*scale)))
            a, b = a.resize(size), b.resize(size)
            row = Image.new("RGB", (size[0]*2, size[1]+28), "white")
            row.paste(a, (0, 28)); row.paste(b, (size[0], 28))
            ImageDraw.Draw(row).text((4, 5), f"PAGE {page_no} #{index} RAW / OUTPUT", fill="black")
            items.append(row)
width = max(item.width for item in items)
height = sum(item.height for item in items)
sheet = Image.new("RGB", (width, height), "#dddddd")
y = 0
for item in items:
    sheet.paste(item, (0, y)); y += item.height
sheet.save(sys.argv[2], quality=90)
