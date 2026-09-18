#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
args = parser.parse_args()
rows = []
for page in sorted((args.job_root / "metadata").glob("page-*")):
    report = page / "typeset-report.json"
    if not report.exists():
        continue
    for item in json.loads(report.read_text(encoding="utf-8")).get("results", []):
        if item.get("status") != "TYPESET":
            continue
        font = int(item.get("font_size", 0))
        preferred = int(item.get("preferred_font_size", font or 1))
        box = item.get("bbox", {})
        rows.append({
            "page": page.name, "id": item.get("id"), "font": font,
            "preferred": preferred, "ratio": round(font / max(1, preferred), 3),
            "width": box.get("width"), "height": box.get("height"),
            "lines": item.get("line_count"), "text": item.get("wrapped"),
        })
print(json.dumps({"count": len(rows), "smallest": sorted(rows, key=lambda x: (x["font"], x["ratio"]))[:30],
                  "under_24": sum(x["font"] < 24 for x in rows),
                  "under_70_percent": sum(x["ratio"] < .7 for x in rows)}, indent=2))

