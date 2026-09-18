#!/usr/bin/env python3
"""Find sparse artwork lettering eligible for exact-mask LaMa, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from remove_text_test import artwork_text_mask, hybrid_artwork_eligible


def scan(job_root: Path) -> dict:
    candidates = []
    pages_scanned = 0
    for page in sorted((job_root / "metadata").glob("page-*")):
        required = [page / "pipeline-status.json", page / "ocr-result.json",
                    page / "translations.json", page / "surface-map.json"]
        if not all(path.is_file() for path in required):
            continue
        status = json.loads(required[0].read_text(encoding="utf-8"))
        input_path = Path(status["input"])
        if not input_path.is_file():
            continue
        with Image.open(input_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        ocr = json.loads(required[1].read_text(encoding="utf-8"))
        translations = json.loads(required[2].read_text(encoding="utf-8"))
        surfaces = json.loads(required[3].read_text(encoding="utf-8"))
        regions = {item["id"]: item for item in ocr.get("regions", [])}
        surface_by_id = {item["id"]: item for item in surfaces.get("surfaces", [])}
        pages_scanned += 1
        for group in translations.get("groups", []):
            surface = surface_by_id.get(group["id"])
            if not surface or surface.get("surface_type") != "ARTWORK_TEXT":
                continue
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            details = []
            valid = True
            for region_id in group.get("source_region_ids", []):
                region = regions.get(region_id)
                if not region:
                    valid = False
                    break
                region_mask, detail = artwork_text_mask(image, region["bbox"])
                details.append({"region_id": region_id, **detail})
                if region_mask is None:
                    valid = False
                    break
                mask = cv2.bitwise_or(mask, region_mask)
            if not valid:
                continue
            eligible = hybrid_artwork_eligible(False, surface, details, mask)
            candidates.append({
                "page": page.name, "group_id": group["id"], "bbox": group["bbox"],
                "text": group.get("text", ""), "translation": group.get("translation", ""),
                "eligible": eligible, "mask_pixels": int(np.count_nonzero(mask)),
                "page_mask_ratio": round(float(np.mean(mask > 0)), 6),
                "maximum_region_mask_ratio": max(
                    (float(item.get("mask_ratio", 1.0)) for item in details), default=1.0
                ),
            })
    return {
        "pages_scanned": pages_scanned, "artwork_regions": len(candidates),
        "eligible_regions": sum(item["eligible"] for item in candidates),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = scan(args.job_root.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
