#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
args = parser.parse_args()
rows = []
for page in sorted((args.job_root / "metadata").glob("page-*")):
    path = page / "quality-report.json"
    if not path.exists():
        continue
    quality = json.loads(path.read_text())
    checks = quality.get("checks", {})
    if quality.get("review_reasons"):
        rows.append({"page": page.name, "reasons": quality["review_reasons"],
                     "outside": checks.get("outside_allowed_pixels", 0),
                     "outside_surface": checks.get("outside_surface_render_pixels", 0),
                     "components": checks.get("outside_components", [])[:3],
                     "residual": checks.get("residual_candidates", [])})
print(json.dumps(rows, indent=2))

