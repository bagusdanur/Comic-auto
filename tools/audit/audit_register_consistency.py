import json
import re
import sys
from pathlib import Path


for job in map(Path, sys.argv[1:]):
    print(job)
    for page in sorted((job / "metadata").glob("page-*")):
        path = page / "translations.json"
        if not path.exists():
            continue
        groups = json.loads(path.read_text(encoding="utf-8")).get("groups", [])
        rows = []
        counts = {"gue_lo": 0, "aku_kamu": 0, "formal": 0}
        for group in groups:
            target = str(group.get("translation", ""))
            register = None
            if re.search(r"\b(?:gue|gua|lo|lu)\b", target, re.I):
                register = "gue_lo"; counts[register] += 1
            if re.search(r"\b(?:aku|kau|kamu)\b", target, re.I):
                register = "aku_kamu"; counts[register] += 1
            if re.search(r"\b(?:saya|anda)\b", target, re.I):
                register = "formal"; counts[register] += 1
            if register:
                rows.append({"id": group["id"], "source": group.get("text"),
                             "target": target, "register": register})
        if counts["gue_lo"] and counts["aku_kamu"]:
            print(json.dumps({"page": page.name, "counts": counts, "rows": rows}, ensure_ascii=False))

