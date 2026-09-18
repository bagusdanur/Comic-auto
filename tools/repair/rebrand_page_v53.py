#!/usr/bin/env python3
"""Reapply current source-brand cleanup to one completed page."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from branding_v53 import apply_branding_v53

parser=argparse.ArgumentParser()
parser.add_argument("output",type=Path)
parser.add_argument("metadata",type=Path)
parser.add_argument("--first-page",action="store_true")
parser.add_argument("--last-page",action="store_true")
args=parser.parse_args()
report=apply_branding_v53(
    args.output,args.metadata/"ocr-result.json",
    ROOT/"assets/branding/watermark.png",
    ROOT/"assets/branding/watermark-text.png",
    first_page=args.first_page,
    banner_path=ROOT/"assets/branding/banner.png" if args.last_page else None,
)
(args.metadata/"branding-report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print(json.dumps(report,ensure_ascii=False,indent=2))
