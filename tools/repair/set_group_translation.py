#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("translations", type=Path)
parser.add_argument("group_id")
parser.add_argument("translation")
args = parser.parse_args()
data = json.loads(args.translations.read_text(encoding="utf-8"))
matches = [item for item in data.get("groups", []) if item.get("id") == args.group_id]
if len(matches) != 1:
    raise SystemExit(f"Expected one group {args.group_id}, found {len(matches)}")
matches[0]["translation"] = args.translation
args.translations.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(args.group_id, "=", args.translation)

