#!/usr/bin/env python3
"""Split four tilted social cards that OCR merged into overlapping groups."""

import argparse
import json
from copy import deepcopy
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("translations", type=Path)
parser.add_argument("ocr", type=Path)
parser.add_argument("removal", type=Path)
args = parser.parse_args()
payload = json.loads(args.translations.read_text(encoding="utf-8"))
ocr = json.loads(args.ocr.read_text(encoding="utf-8"))
regions = {item["id"]: item for item in ocr["regions"]}


def box(ids):
    items = [regions[item]["bbox"] for item in ids]
    x1, y1 = min(item["x"] for item in items), min(item["y"] for item in items)
    x2 = max(item["x"] + item["width"] for item in items)
    y2 = max(item["y"] + item["height"] for item in items)
    return {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


cards = [
    ("bubble-013", ["14d747fc-b828-4175-9b88-29fd34acccb3", "2d4e9106-a401-4b42-ada3-36466f4cc9be"], "Artis memujinya..."),
    ("bubble-014", ["485a7a54-f849-4550-adf0-5c19e379404f", "e4f4b671-1f73-40ac-a4c0-fa972a6f4f10", "46823ca7-7626-4ca2-b3ed-d48fe3ea8fae", "5be86c64-0929-4653-bb4c-c2d73a60f620"], "Ah, mau kusimpan sendiri. Siapa yang menyebarkannya?"),
    ("bubble-015", ["a343a0a4-9e65-45bc-9b5a-84739398d335", "daaecbb3-1baf-484b-a39c-b7c51f6a3aec"], "Gak nyangka ada stroberi seperti itu di dunia ini..."),
    ("bubble-015b", ["db92d739-48a5-4396-9c8e-ccd9f9b75018", "5be1e51f-440e-4d2f-bbd2-92224fc83fc3"], "Lihat warnanya. Kelihatan enak banget. T_T"),
]
old = {item["id"]: item for item in payload["groups"]}
replacement = []
for group_id, ids, translation in cards:
    base = deepcopy(old["bubble-015" if group_id == "bubble-015b" else group_id])
    base.update({"id": group_id, "text": " ".join(regions[item]["text"] for item in ids), "translation": translation, "source_region_ids": ids, "bbox": box(ids), "style": "dialogue"})
    replacement.append(base)
payload["groups"] = [item for item in payload["groups"] if item["id"] not in {"bubble-013", "bubble-014", "bubble-015"}] + replacement
payload["groups"].sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
args.translations.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

removal = json.loads(args.removal.read_text(encoding="utf-8"))
by_id = {item["id"]: item for item in removal["results"]}
if "bubble-015b" not in by_id:
    clone = deepcopy(by_id["bubble-015"])
    clone["id"] = "bubble-015b"
    removal["results"].append(clone)
args.removal.write_text(json.dumps(removal, ensure_ascii=False, indent=2), encoding="utf-8")
print("social_cards=4")

