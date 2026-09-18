#!/usr/bin/env python3
"""Merge OCR groups that split one visual sentence into overlapping fragments."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("translations", type=Path)
parser.add_argument("first_id")
parser.add_argument("second_id")
parser.add_argument("--translation", required=True)
args = parser.parse_args()
payload = json.loads(args.translations.read_text(encoding="utf-8"))
groups = payload["groups"]
first = next(item for item in groups if item["id"] == args.first_id)
second = next(item for item in groups if item["id"] == args.second_id)
second["text"] = f'{first["text"]} {second["text"]}'.strip()
second["translation"] = args.translation.strip()
payload["groups"] = [item for item in groups if item["id"] != args.first_id]
args.translations.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"merged={args.first_id}+{args.second_id} groups={len(payload['groups'])}")

