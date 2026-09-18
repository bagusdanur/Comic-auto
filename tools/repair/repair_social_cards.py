#!/usr/bin/env python3
"""Deterministically repair legacy tilted white-card captions in one page area."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from typeset_test import fit_text, order_quad


parser = argparse.ArgumentParser()
parser.add_argument("base", type=Path, help="Existing translated page")
parser.add_argument("original", type=Path)
parser.add_argument("translations", type=Path)
parser.add_argument("surface_map", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--restore-y1", type=int)
parser.add_argument("--restore-y2", type=int)
parser.add_argument("--ids", required=True)
parser.add_argument("--font", default="assets/fonts/animeace/animeace2bb_tt/animeace2_reg.ttf")
args = parser.parse_args()

base = np.asarray(Image.open(args.base).convert("RGB")).copy()
original = np.asarray(Image.open(args.original).convert("RGB"))
if base.shape != original.shape:
    raise SystemExit("dimension mismatch")
if args.restore_y1 is not None or args.restore_y2 is not None:
    if args.restore_y1 is None or args.restore_y2 is None:
        raise SystemExit("restore-y1 and restore-y2 must be supplied together")
    base[args.restore_y1:args.restore_y2] = original[args.restore_y1:args.restore_y2]
groups = {item["id"]: item for item in json.loads(args.translations.read_text(encoding="utf-8"))["groups"]}
surfaces = {item["id"]: item for item in json.loads(args.surface_map.read_text(encoding="utf-8"))["surfaces"]}
target_ids = [value.strip() for value in args.ids.split(",") if value.strip()]
changed_allowed = np.zeros(base.shape[:2], dtype=np.uint8)

operations = []
for group_id in target_ids:
    group, surface = groups[group_id], surfaces[group_id]
    quad = order_quad(np.asarray(surface["raw_polygon"], dtype=np.float32))
    # A two-pixel expansion removes antialiasing while remaining inside the
    # already-white card surrounding the RAW lettering.
    center = np.mean(quad, axis=0)
    cleanup_quad = center + (quad - center) * 1.035
    cv2.fillConvexPoly(base, np.round(cleanup_quad).astype(np.int32), (255, 255, 255))
    cv2.fillConvexPoly(changed_allowed, np.round(cleanup_quad).astype(np.int32), 255)
    operations.append((group_id, group, quad, cleanup_quad))

for group_id, group, quad, cleanup_quad in operations:
    top = np.linalg.norm(quad[1] - quad[0])
    bottom = np.linalg.norm(quad[2] - quad[3])
    left = np.linalg.norm(quad[3] - quad[0])
    right = np.linalg.norm(quad[2] - quad[1])
    width = max(8, round(max(top, bottom)))
    height = max(8, round(max(left, right)))
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    fitted = fit_text(draw, group["translation"].upper(), args.font,
                      max(1, round(width * 0.90)), max(1, round(height * 0.84)),
                      9, min(28, max(14, round(height * 0.34))), 0.14, 0)
    if not fitted:
        raise SystemExit(f"text overflow: {group_id}")
    bbox = fitted["text_bbox"]
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(((width - text_width) / 2 - bbox[0], (height - text_height) / 2 - bbox[1]),
                        fitted["wrapped"], font=fitted["font"], fill=(12, 12, 12, 255),
                        spacing=fitted["spacing"], align="center")
    source_quad = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(source_quad, quad.astype(np.float32))
    warped = cv2.warpPerspective(np.asarray(layer), matrix, (base.shape[1], base.shape[0]),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
    alpha = warped[:, :, 3:4].astype(np.float32) / 255
    base = np.round(base * (1 - alpha) + warped[:, :, :3] * alpha).astype(np.uint8)
    cv2.fillConvexPoly(changed_allowed, np.round(cleanup_quad).astype(np.int32), 255)

args.output.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(base).save(args.output, format="PNG", optimize=True)
mask_path = args.output.with_suffix(".mask.png")
Image.fromarray(changed_allowed).save(mask_path, format="PNG", optimize=True)
print(json.dumps({"output": str(args.output), "mask": str(mask_path), "ids": target_ids}, indent=2))

