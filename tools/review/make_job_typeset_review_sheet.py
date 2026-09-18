#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--final", type=Path, help="Optional replacement final image for page 1")
args = parser.parse_args()

image_exts = {".png", ".jpg", ".jpeg", ".webp"}
originals = sorted(p for p in (args.job_root / "original").iterdir() if p.suffix.lower() in image_exts)
outputs = sorted(p for p in (args.job_root / "output").iterdir() if p.suffix.lower() in image_exts)
rows = []

for page_dir in sorted((args.job_root / "metadata").glob("page-*")):
    report_path = page_dir / "typeset-report.json"
    if not report_path.exists():
        continue
    page_index = int(page_dir.name.split("-")[-1]) - 1
    if page_index >= len(originals) or page_index >= len(outputs):
        continue
    original = Image.open(originals[page_index]).convert("RGB")
    final_path = args.final if args.final and page_index == 0 else outputs[page_index]
    final = Image.open(final_path).convert("RGB")
    for item in json.loads(report_path.read_text(encoding="utf-8")).get("results", []):
        if item.get("status") != "NEEDS_REVIEW":
            continue
        box = item["bbox"]
        margin = 55
        crop = (
            max(0, box["x"] - margin), max(0, box["y"] - margin),
            min(original.width, box["x"] + box["width"] + margin),
            min(original.height, box["y"] + box["height"] + margin),
        )
        images = [source.crop(crop) for source in (original, final)]
        scale = min(2.0, 650 / max(1, images[0].width))
        size = (round(images[0].width * scale), round(images[0].height * scale))
        rows.append((page_dir.name, item, [image.resize(size) for image in images]))

row_heights = [images[0].height + 56 for _, _, images in rows]
sheet = Image.new("RGB", (1340, max(1, sum(row_heights))), "#181c24")
draw = ImageDraw.Draw(sheet)
y = 0
for (page, item, images), height in zip(rows, row_heights):
    title = f"{page} {item['id']} {item.get('reason', '')}"
    draw.text((10, y + 7), title, fill="white")
    for x, label, image in zip((10, 680), ("RAW", "FINAL"), images):
        draw.text((x, y + 27), label, fill="#61dafb")
        sheet.paste(image, (x, y + 45))
    y += height

args.output.parent.mkdir(parents=True, exist_ok=True)
sheet.save(args.output, optimize=True)
print(f"wrote {args.output} with {len(rows)} review regions")

