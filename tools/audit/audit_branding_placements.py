#!/usr/bin/env python3
"""Summarize replacement-logo sizes and verify they avoid protected OCR boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from brand_postprocess import boxes_overlap, credit_regions, substantially_same_area


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    args = parser.parse_args()
    failures = 0
    for page in sorted((args.job_dir / "metadata").glob("page-*")):
        report = json.loads((page / "branding-report.json").read_text(encoding="utf-8"))
        ocr = json.loads((page / "ocr-result.json").read_text(encoding="utf-8"))
        with Image.open(page / "typeset.png") as image:
            width = image.width
        credits = credit_regions(ocr)
        protected = [
            item for item in ocr.get("regions", []) if item.get("bbox")
            and not any(substantially_same_area(item["bbox"], credit["bbox"]) for credit in credits)
        ]
        for replacement in report.get("replacements", []):
            logo = replacement.get("logo")
            if not logo:
                continue
            x, y, logo_width, logo_height = logo
            colliders = [
                item.get("text", "") for item in protected
                if boxes_overlap((x, y, x + logo_width, y + logo_height), item["bbox"], 0)
            ]
            collision = bool(colliders)
            oversized = logo_width > round(width * 0.20)
            if collision or oversized:
                failures += 1
                print(page.name, replacement.get("mode"), logo, f"ratio={logo_width / width:.3f}",
                      f"collision={collision}", f"oversized={oversized}")
                print("  colliders=", colliders)
    print(f"branding_placement_failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
