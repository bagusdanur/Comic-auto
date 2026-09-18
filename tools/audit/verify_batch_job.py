#!/usr/bin/env python3
import argparse
import json
import zipfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
parser.add_argument("--page", default="page-001")
args = parser.parse_args()
with zipfile.ZipFile(args.job_root / "output/chapter-translated.zip") as archive:
    names = archive.namelist()
    image_count = len([name for name in names
                       if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))])
pipeline = json.loads((args.job_root / "metadata" / args.page / "pipeline-status.json").read_text())
print(json.dumps({
    "zip_images": image_count, "all_webp": all(name.lower().endswith(".webp") for name in names),
    "archive_bytes": (args.job_root / "output/chapter-translated.zip").stat().st_size,
    "first_members": names[:5], "page": args.page, "page_status": pipeline.get("status"),
    "passthrough": pipeline.get("passthrough", False), "review_ids": pipeline.get("review_ids", []),
}, indent=2))

