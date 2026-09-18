import json
import zipfile
from pathlib import Path

from PIL import Image

from brand_postprocess import alpha_composite_clipped, inpaint_old_mark, resized

root = Path("/tmp/ch29-brand-final")
with Image.open("assets/branding/watermark.png") as source:
    logo = source.convert("RGBA")
    bbox = logo.getchannel("A").getbbox()
    if bbox:
        logo = logo.crop(bbox)

for report_path in sorted(root.glob("[0-9][0-9][0-9]-branding.json")):
    page_path = root / f"{report_path.stem[:3]}.png"
    report = json.loads(report_path.read_text())
    with Image.open(page_path) as source:
        image = source.convert("RGBA")
    for replacement in report.get("replacements", []):
        box = replacement.get("covered") or replacement.get("cleared")
        if not box:
            continue
        x1, y1, x2, y2 = map(int, box)
        image = inpaint_old_mark(image, (x1, y1, x2, y2))
        target_width = min(round(image.width * 0.58), max(round(image.width * 0.38), round((x2 - x1) * 1.20)))
        mark = resized(logo, target_width)
        px = max(0, min(image.width - mark.width, x1 + ((x2 - x1) - mark.width) // 2))
        py = max(0, min(image.height - mark.height, y1 + ((y2 - y1) - mark.height) // 2))
        alpha_composite_clipped(image, mark, px, py)
    image.convert("RGB").save(page_path, format="PNG", optimize=True)

with zipfile.ZipFile("/tmp/chapter-translated-v7-logo-clean.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for page in sorted(root.glob("[0-9][0-9][0-9].png")):
        archive.write(page, page.name)

