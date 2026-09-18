#!/usr/bin/env python3
"""Bounded exact-mask LaMa smoke test; writes only to an explicit output."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from xianscan_client import XianScanClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8123")
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    with Image.open(args.image) as opened:
        image = np.asarray(opened.convert("RGB"))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    box_width = min(80, max(16, image.shape[1] // 10))
    box_height = min(48, max(12, image.shape[0] // 100))
    x1, y1 = max(0, image.shape[1] // 2 - box_width // 2), max(0, image.shape[0] // 20)
    mask[y1:y1 + box_height, x1:x1 + box_width] = 255
    started = time.perf_counter()
    result = XianScanClient(args.url, timeout=args.timeout).inpaint_mask(image, mask)
    seconds = time.perf_counter() - started
    changed = np.any(result != image, axis=2)
    outside = int(np.count_nonzero(changed & (mask == 0)))
    if outside:
        raise RuntimeError(f"Smoke result changed {outside} pixels outside mask")
    Image.fromarray(result).save(args.output, format="PNG")
    print(json.dumps({
        "seconds": round(seconds, 3), "mask_pixels": int(np.count_nonzero(mask)),
        "changed_pixels": int(np.count_nonzero(changed)), "outside_mask_changed_pixels": outside,
        "dimensions": [image.shape[1], image.shape[0]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
