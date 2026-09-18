#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_root", type=Path)
args = parser.parse_args()
details = []
for page in sorted((args.job_root / "metadata").glob("page-*")):
    report_path = page / "typeset-report.json"
    translations_path = page / "translations.json"
    if not report_path.exists() or not translations_path.exists():
        continue
    groups = {item["id"]: item for item in json.loads(translations_path.read_text()).get("groups", [])}
    for item in json.loads(report_path.read_text()).get("results", []):
        if item.get("status") == "NEEDS_REVIEW":
            group = groups.get(item.get("id"), {})
            details.append({"page": page.name, "id": item.get("id"), "reason": item.get("reason"),
                            "font": item.get("font_size"), "preferred": item.get("preferred_font_size"),
                            "style": group.get("style"), "source": group.get("text"),
                            "translation": group.get("translation"), "bbox": item.get("bbox")})
print(json.dumps(details, ensure_ascii=False, indent=2))

