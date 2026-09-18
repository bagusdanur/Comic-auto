#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from quality_pipeline_v2 import validate

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
parser.add_argument("--manual-pass-page", action="append", default=[])
args = parser.parse_args()

for page in sorted((args.job_root / "metadata").glob("page-*")):
    quality_path = page / "quality-report.json"
    pipeline_path = page / "pipeline-status.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("passthrough"):
        continue
    if page.name in args.manual_pass_page:
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        quality.update({"status": "PASS", "review_reasons": [],
                        "manual_override": "TARGETED_FULL_CAPTION_VISUAL_QA"})
        quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    else:
        validate(
            Path(json.loads((page / "pipeline-status.json").read_text(encoding="utf-8"))["input"]),
            page / "cleaned.png", page / "typeset.png", page / "mask-debug.png",
            page / "surface-map.json", quality_path,
        )
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    typeset = json.loads((page / "typeset-report.json").read_text(encoding="utf-8"))
    review_ids = sorted(item["id"] for item in typeset.get("results", [])
                        if item.get("status") == "NEEDS_REVIEW")
    if quality.get("status") != "PASS":
        review_ids.extend(f"quality:{reason}" for reason in quality.get("review_reasons", []))
    pipeline["review_ids"] = sorted(set(review_ids))
    pipeline["quality_v2"] = quality
    pipeline["status"] = "REVIEW" if review_ids else "COMPLETED"
    pipeline_path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
print("revalidated", len(list((args.job_root / "metadata").glob("page-*"))), "pages")
