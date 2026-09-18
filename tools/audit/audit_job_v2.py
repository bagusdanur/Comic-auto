#!/usr/bin/env python3
import argparse
import collections
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
args = parser.parse_args()

summary = {"pages": 0, "quality_reports": 0, "pipeline_statuses": collections.Counter(),
           "typeset_reviews": [], "quality_reasons": collections.Counter(), "quality_pages": []}
for page in sorted((args.job_root / "metadata").glob("page-*")):
    summary["pages"] += 1
    status_path = page / "pipeline-status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        summary["pipeline_statuses"][status.get("status", "MISSING")] += 1
    typeset_path = page / "typeset-report.json"
    if typeset_path.exists():
        for item in json.loads(typeset_path.read_text(encoding="utf-8")).get("results", []):
            if item.get("status") == "NEEDS_REVIEW":
                summary["typeset_reviews"].append({"page": page.name, "id": item.get("id"),
                                                   "reason": item.get("reason"), "bbox": item.get("bbox")})
    quality_path = page / "quality-report.json"
    if quality_path.exists():
        summary["quality_reports"] += 1
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        reasons = quality.get("review_reasons", [])
        for reason in reasons:
            summary["quality_reasons"][reason] += 1
        if reasons:
            summary["quality_pages"].append({"page": page.name, "status": quality.get("status"),
                                             "reasons": reasons})
summary["pipeline_statuses"] = dict(summary["pipeline_statuses"])
summary["quality_reasons"] = dict(summary["quality_reasons"])
print(json.dumps(summary, indent=2))

