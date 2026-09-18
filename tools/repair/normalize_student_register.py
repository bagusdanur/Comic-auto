import json
import sys
from pathlib import Path


canonical = {
    "LIKE THIS BITCH?": "Lo suka sama jalang ini?",
    "DO YOU LIKE THIS BITCH?": "Lo suka sama jalang ini?",
    "YOU'VE GOT SOME NERVE.": "Nyali lo gede juga.",
    "HOW DARE YOU LIKE HER... THE ONE BROTHER DAEHAN PERSONALLY REGARDED AS A SLAVE? HAHA.":
        "Berani-beraninya lo suka sama dia... padahal Bang Daehan sendiri menganggapnya budak? Haha.",
    "I'M NOT DONE YET.": "Gue belum selesai.",
}

for job in map(Path, sys.argv[1:]):
    for path in sorted((job / "metadata").glob("page-*/translations.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = []
        for group in data.get("groups", []):
            target = canonical.get(" ".join(str(group.get("text", "")).upper().split()))
            if target and group.get("translation") != target:
                group["translation"] = target
                changed.append(group["id"])
        if changed:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(path.parent.name, ",".join(changed))
