import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
group_id = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
matches = [item for item in data.get("surfaces", []) if item.get("id") == group_id]
if len(matches) != 1:
    raise SystemExit(f"Expected one surface {group_id}, found {len(matches)}")
matches[0]["source_equals_translation"] = True
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(group_id, "source_equals_translation=true")

