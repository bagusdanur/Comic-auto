import json
import re
import sys
from pathlib import Path


# Pages dominated by family/adult/formal speech. Everything else in these four
# chapters is a school/peer/gang scene and uses the title's aku/kau register.
excluded = {
    "10": set(),
    "11": set(),
    "12": {"page-005", "page-011", "page-013", "page-021", "page-022"},
    "13": {"page-002"},
}


def naturalize(text: str) -> str:
    value = re.sub(r"\b(?:lo|lu|kamu|kau)\b", "kau", text, flags=re.I)
    value = re.sub(r"\b(?:gue|gua|aku)\b", "aku", value, flags=re.I)
    possessives = {
        "tangan gue": "tanganku", "tenggorokan gue": "tenggorokanku",
        "perasaan gue": "perasaanku", "keluarga gue": "keluargaku",
        "keinginan gue": "keinginanku", "bantuan lo": "bantuanmu",
        "kesehatan lo": "kesehatanmu", "nilai lo": "nilaimu",
    }
    for source, target in possessives.items():
        value = re.sub(rf"\b{source}\b", target, value, flags=re.I)
    return value


for job in map(Path, sys.argv[1:]):
    chapter = job.parent.name.removeprefix("chapter-")
    for path in sorted((job / "metadata").glob("page-*/translations.json")):
        if path.parent.name in excluded.get(chapter, set()):
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        changed = []
        for group in data.get("groups", []):
            # Apply the series register to every translated text kind, including
            # inner monologue/captions. Previously these could leak gue/lo even
            # when all speech bubbles had already been normalized.
            old = str(group.get("translation", ""))
            new = naturalize(old)
            if new != old:
                group["translation"] = new
                changed.append(group["id"])
        if changed:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(chapter, path.parent.name, ",".join(changed))
