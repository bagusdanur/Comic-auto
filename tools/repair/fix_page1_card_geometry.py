#!/usr/bin/env python3
"""Repair the one legacy page-1 social-card group produced before shape-aware OCR."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("translations", type=Path)
args = parser.parse_args()
payload = json.loads(args.translations.read_text(encoding="utf-8"))
for group in payload.get("groups", []):
    if group.get("id") == "bubble-014":
        group["bbox"] = {"x": 230, "y": 7635, "width": 400, "height": 165}
args.translations.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

