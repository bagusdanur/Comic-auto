#!/usr/bin/env python3
"""Apply explicit group translation overrides from a JSON mapping."""

import json
import sys
from pathlib import Path


job = Path(sys.argv[1])
patches = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
for page_name, overrides in patches.items():
    path = job / "metadata" / page_name / "translations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for group in payload.get("groups", []):
        if group.get("id") in overrides:
            group["translation"] = overrides[group["id"]]
            changed += 1
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{page_name}: patched {changed} translation(s)")

