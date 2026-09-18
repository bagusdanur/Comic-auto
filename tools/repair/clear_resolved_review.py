#!/usr/bin/env python3
"""Mark explicitly repaired legacy review regions as resolved."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--ids", required=True)
args = parser.parse_args()
resolved = {value.strip() for value in args.ids.split(",") if value.strip()}
payload = json.loads(args.report.read_text(encoding="utf-8"))
for item in payload.get("results", []):
    if item.get("id") in resolved and item.get("status") == "NEEDS_REVIEW":
        item["status"] = "RESOLVED_MANUALLY"
        item["reason"] = "TARGETED_RAW_REPAIR"
args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

