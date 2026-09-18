#!/usr/bin/env python3
"""List relevant model IDs from an OpenAI-compatible provider safely."""

import argparse
import json
import urllib.request
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("env", type=Path)
parser.add_argument("--filter", default="5.6,image,vision,gemini")
args = parser.parse_args()
values = {}
for line in args.env.read_text(encoding="utf-8").splitlines():
    if "=" in line and not line.lstrip().startswith("#"):
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
base = values["NINEROUTER_BASE_URL"].rstrip("/")
request = urllib.request.Request(base + "/models", headers={"Authorization": "Bearer " + values["NINEROUTER_API_KEY"]})
with urllib.request.urlopen(request, timeout=30) as response:
    payload = json.load(response)
needles = [item.strip().casefold() for item in args.filter.split(",")]
for model in sorted(item.get("id", "") for item in payload.get("data", [])):
    if any(needle in model.casefold() for needle in needles):
        print(model)

