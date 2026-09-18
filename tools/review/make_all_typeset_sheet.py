#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--from-page", type=int, default=1)
parser.add_argument("--to-page", type=int, default=9999)
parser.add_argument("--pages", help="Comma-separated page numbers")
parser.add_argument("--max-regions", type=int, default=0)
args = parser.parse_args()
selected_pages={int(value) for value in args.pages.split(',')} if args.pages else None
exts = {".png", ".jpg", ".jpeg", ".webp"}
raws = sorted(p for p in (args.job_root / "original").iterdir() if p.suffix.lower() in exts)
finals = sorted(p for p in (args.job_root / "output").iterdir() if p.suffix.lower() in exts)
rows = []
for page in sorted((args.job_root / "metadata").glob("page-*")):
    index = int(page.name.split("-")[-1]) - 1
    if selected_pages is not None and index + 1 not in selected_pages:
        continue
    if not (args.from_page <= index + 1 <= args.to_page):
        continue
    report = page / "typeset-report.json"
    if not report.exists() or index >= len(raws) or index >= len(finals):
        continue
    # The published output may be vertically cropped when a source banner is
    # removed. Typeset report coordinates are pre-branding, so compare against
    # the matching pre-branding render to avoid blank/misaligned final crops.
    typeset_image = page / "typeset.png"
    final_path = typeset_image if typeset_image.exists() else finals[index]
    raw, final = Image.open(raws[index]).convert("RGB"), Image.open(final_path).convert("RGB")
    for item in json.loads(report.read_text())["results"]:
        if item.get("status") != "TYPESET":
            continue
        box = item["bbox"]
        margin = max(35, round(max(box["width"], box["height"]) * .12))
        crop = (max(0, box["x"]-margin), max(0, box["y"]-margin),
                min(raw.width, box["x"]+box["width"]+margin),
                min(raw.height, box["y"]+box["height"]+margin))
        pair = [raw.crop(crop), final.crop(crop)]
        scale = min(1.35, 300/max(1, pair[0].width))
        size = (round(pair[0].width*scale), round(pair[0].height*scale))
        font=Path(str(item.get('font',''))).stem
        label=f"{page.name} {item['id']} | {item.get('font_variant','regular')} | {font}"
        rows.append((label, [x.resize(size) for x in pair]))
        if args.max_regions and len(rows)>=args.max_regions:
            break
    if args.max_regions and len(rows)>=args.max_regions:
        break
cell_w = 315
heights = [max(x.height for x in pair)+34 for _, pair in rows]
sheet = Image.new("RGB", (cell_w*2, max(1,sum(heights))), "#151923")
draw = ImageDraw.Draw(sheet)
y=0
for (label,pair),height in zip(rows,heights):
    draw.text((4,y+4), label, fill="white")
    sheet.paste(pair[0],(4,y+24)); sheet.paste(pair[1],(cell_w+4,y+24)); y+=height
sheet.save(args.output, quality=88)
print(f"wrote {len(rows)} regions")

