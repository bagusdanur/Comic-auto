#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
pipeline = args.job_root / "metadata/page-001"
original_path = next((args.job_root / "original").iterdir())
final_path = next(path for path in (args.job_root / "output").iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
original = Image.open(original_path).convert("RGB")
cleaned = Image.open(pipeline / "cleaned.png").convert("RGB")
final = Image.open(final_path).convert("RGB")
groups = {item["id"]: item for item in json.loads((pipeline / "translations.json").read_text())["groups"]}
results = json.loads((pipeline / "removal-report.json").read_text())["results"]
selected = [item for item in results if item["status"] in {"NEEDS_REVIEW", "INPAINTED"}]
rows = []
for item in selected:
    box = groups[item["id"]]["bbox"]
    margin = 45
    crop = (
        max(0, box["x"] - margin), max(0, box["y"] - margin),
        min(original.width, box["x"] + box["width"] + margin),
        min(original.height, box["y"] + box["height"] + margin),
    )
    images = [source.crop(crop) for source in (original, cleaned, final)]
    scale = min(2.5, 520 / max(1, images[0].width))
    size = (round(images[0].width * scale), round(images[0].height * scale))
    rows.append((item, [image.resize(size) for image in images]))
row_heights = [images[0].height + 55 for _, images in rows]
sheet = Image.new("RGB", (1590, max(1, sum(row_heights))), "#202020")
draw = ImageDraw.Draw(sheet)
y = 0
for (item, images), height in zip(rows, row_heights):
    reason = ",".join(str(region.get("reason", "")) for region in item.get("regions", []))
    draw.text((8, y + 5), f"{item['id']} {item['status']} {reason}", fill="white")
    for x, label, image in zip((8, 535, 1062), ("ORIGINAL", "CLEANED", "FINAL"), images):
        draw.text((x, y + 25), label, fill="#8ec5ff")
        sheet.paste(image, (x, y + 43))
    y += height
args.output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.output, optimize=True)

