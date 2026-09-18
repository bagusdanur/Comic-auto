#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

originals = sorted(path for path in (args.job_root / "original").iterdir() if path.is_file())
finals = sorted(path for path in (args.job_root / "output").iterdir() if path.is_file())
rows = []
for page_index, metadata in enumerate(sorted((args.job_root / "metadata").glob("page-*"))):
    quality_path = metadata / "quality-report.json"
    if not quality_path.exists() or page_index >= len(originals) or page_index >= len(finals):
        continue
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not quality.get("review_reasons"):
        continue
    groups = {item["id"]: item for item in json.loads((metadata / "translations.json").read_text(encoding="utf-8"))["groups"]}
    checks = quality.get("checks", {})
    boxes = []
    for item in checks.get("residual_candidates", []):
        if item.get("id") in groups:
            boxes.append(groups[item["id"]]["bbox"])
    boxes.extend(checks.get("outside_components", [])[:3])
    if not boxes:
        continue
    x1 = max(0, min(int(box["x"]) for box in boxes) - 60)
    y1 = max(0, min(int(box["y"]) for box in boxes) - 60)
    x2 = max(int(box["x"]) + int(box["width"]) for box in boxes) + 60
    y2 = max(int(box["y"]) + int(box["height"]) for box in boxes) + 60
    raw = Image.open(originals[page_index]).convert("RGB")
    final = Image.open(finals[page_index]).convert("RGB")
    crop = (x1, y1, min(raw.width, x2), min(raw.height, y2))
    pair = [raw.crop(crop), final.crop(crop)]
    scale = min(2.0, 520 / max(1, pair[0].width))
    size = (round(pair[0].width * scale), round(pair[0].height * scale))
    rows.append((metadata.name, ", ".join(quality["review_reasons"]), [item.resize(size) for item in pair]))

row_heights = [images[0].height + 48 for _, _, images in rows]
sheet = Image.new("RGB", (1080, max(1, sum(row_heights))), "#10151f")
draw = ImageDraw.Draw(sheet)
y = 0
for (page, reasons, images), row_height in zip(rows, row_heights):
    draw.text((8, y + 5), f"{page}: {reasons}", fill="white")
    draw.text((8, y + 24), "RAW", fill="#7dd3fc")
    draw.text((548, y + 24), "FINAL", fill="#86efac")
    sheet.paste(images[0], (8, y + 42))
    sheet.paste(images[1], (548, y + 42))
    y += row_height
args.output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.output, quality=92)
print(f"wrote {args.output} with {len(rows)} pages")

