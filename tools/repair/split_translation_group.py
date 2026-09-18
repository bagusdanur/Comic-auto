#!/usr/bin/env python3
"""Split one wrongly merged translation group at a source-region boundary."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("translations", type=Path)
parser.add_argument("ocr", type=Path)
parser.add_argument("group_id")
parser.add_argument("split_before_region_id")
parser.add_argument("--first-translation", required=True)
parser.add_argument("--second-translation", required=True)
parser.add_argument("--drop-region", action="append", default=[])
args = parser.parse_args()

payload = json.loads(args.translations.read_text(encoding="utf-8"))
ocr = json.loads(args.ocr.read_text(encoding="utf-8"))
regions = {item["id"]: item for item in ocr["regions"]}
index = next(i for i, item in enumerate(payload["groups"]) if item["id"] == args.group_id)
group = payload["groups"][index]
ids = [item for item in group["source_region_ids"] if item not in set(args.drop_region)]
split = ids.index(args.split_before_region_id)
parts = [ids[:split], ids[split:]]
if any(not part for part in parts):
    raise SystemExit("Split must leave at least one source region on each side")


def build(part_ids, translation):
    selected = [regions[item] for item in part_ids]
    left = min(item["bbox"]["x"] for item in selected)
    top = min(item["bbox"]["y"] for item in selected)
    right = max(item["bbox"]["x"] + item["bbox"]["width"] for item in selected)
    bottom = max(item["bbox"]["y"] + item["bbox"]["height"] for item in selected)
    return {
        "id": "", "text": " ".join(item["text"] for item in selected),
        "bbox": {"x": left, "y": top, "width": right - left, "height": bottom - top},
        "source_region_ids": part_ids, "translation": translation, "style": group.get("style", "narration"),
    }


replacement = [
    build(parts[0], args.first_translation.strip()),
    build(parts[1], args.second_translation.strip()),
]
payload["groups"][index:index + 1] = replacement
for number, item in enumerate(payload["groups"], 1):
    item["id"] = f"bubble-{number:03d}"
args.translations.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(replacement, ensure_ascii=False, indent=2))

