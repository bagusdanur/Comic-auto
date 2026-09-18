import json
import shutil
import zipfile
from pathlib import Path

from brand_postprocess import apply_branding

source = Path("/tmp/ch29-font-final")
destination = Path("/tmp/ch29-brand-final")
job = Path("data/jobs/library/you-like-someone-with-that-face/chapter-29/2a6187df-7202-457e-88f8-e42230065d8a")
destination.mkdir(parents=True, exist_ok=True)
pages = sorted(source.glob("[0-9][0-9][0-9].png"))
for index, page in enumerate(pages):
    target = destination / page.name
    shutil.copy2(page, target)
    report = apply_branding(
        target,
        job / "metadata" / f"page-{index + 1:03d}" / "ocr-result.json",
        Path("assets/branding/watermark.png"),
        Path("assets/branding/watermark-text.png"),
        first_page=index == 0,
        banner_path=Path("assets/branding/banner.png") if index == len(pages) - 1 else None,
    )
    (destination / f"{index + 1:03d}-branding.json").write_text(json.dumps(report, indent=2))
    print(index + 1, report)

with zipfile.ZipFile("/tmp/chapter-translated-v5-brand-final.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for page in sorted(destination.glob("[0-9][0-9][0-9].png")):
        archive.write(page, page.name)

