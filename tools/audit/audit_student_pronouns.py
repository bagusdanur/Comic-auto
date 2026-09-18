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
        rows = []
        for group in json.loads(path.read_text(encoding="utf-8")).get("groups", []):
            target = str(group.get("translation", ""))
            if group.get("style") in {"dialogue", "shout", "chat"} and re.search(
                r"\b(?:aku|kau|kamu|gue|gua|lo|lu)\b", target, re.I
            ):
                rows.append({"id": group["id"], "source": group.get("text"),
                             "target": target, "style": group.get("style")})
        if rows:
            print(json.dumps({"page": page.name, "rows": rows}, ensure_ascii=False))

