#!/usr/bin/env python3
"""Remove known source-site promo groups from cached translation metadata."""

import json
import re
import sys
from pathlib import Path


PROMO_RE = re.compile(
    r"(?:discover\s+more\s+chapters?|discord|read\b.{0,30}\b(?:site|website)|"
    r"lightning[-\s]*fast\s+releases?|premium\s+quality\s+(?:thriving\s+)?community)",
    re.IGNORECASE,
)


for job_arg in sys.argv[1:]:
    job = Path(job_arg)
    for path in sorted((job / "metadata").glob("page-*/translations.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups = payload.get("groups", [])
        kept = [item for item in groups if not PROMO_RE.search(str(item.get("text", "")))]
        if len(kept) == len(groups):
            continue
        payload["groups"] = kept
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{path.parent.name}: removed {len(groups) - len(kept)} source promo group(s)")

