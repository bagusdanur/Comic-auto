import json
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "metadata/page-017/translations.json"
data = json.loads(path.read_text(encoding="utf-8"))
replacements = {
    "bubble-003": "Pamer urat lengan.",
    "bubble-006": "Kucai Daejeon",
    "bubble-009": "Roti Manis",
    "bubble-014": "Kucai",
    "bubble-015": "Roti Manis",
}
for group in data.get("groups", []):
    if group.get("id") in replacements:
        group["translation"] = replacements[group["id"]]
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(replacements, ensure_ascii=False))

