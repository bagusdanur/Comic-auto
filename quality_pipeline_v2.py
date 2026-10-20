#!/usr/bin/env python3
"""Lightweight surface analysis and hard post-render quality checks.

This module never edits comic pixels. It creates deterministic geometry used by
cleanup/typesetting and validates a rendered page. Expensive AI escalation is
intentionally outside the fast path.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    result = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    result[0], result[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    result[1], result[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return result


def source_angle(regions: list[dict]) -> float:
    values = []
    for region in regions:
        polygon = region.get("polygon") or []
        if len(polygon) >= 2:
            dx = polygon[1][0] - polygon[0][0]
            dy = polygon[1][1] - polygon[0][1]
            if dx:
                values.append(math.degrees(math.atan2(dy, dx)))
    return round(float(np.median(values)), 2) if values else 0.0


def raw_quad(regions: list[dict], bbox: dict, angle: float) -> np.ndarray:
    points = [point for region in regions for point in (region.get("polygon") or [])]
    if not points:
        x, y = int(bbox["x"]), int(bbox["y"])
        width, height = int(bbox["width"]), int(bbox["height"])
        return np.float32([[x, y], [x + width, y], [x + width, y + height], [x, y + height]])
    radians = math.radians(angle)
    axis_x = np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)
    axis_y = np.array([-math.sin(radians), math.cos(radians)], dtype=np.float32)
    cloud = np.asarray(points, dtype=np.float32)
    projected_x, projected_y = cloud @ axis_x, cloud @ axis_y
    min_x, max_x = projected_x.min(), projected_x.max()
    min_y, max_y = projected_y.min(), projected_y.max()
    return np.float32([
        axis_x * min_x + axis_y * min_y,
        axis_x * max_x + axis_y * min_y,
        axis_x * max_x + axis_y * max_y,
        axis_x * min_x + axis_y * max_y,
    ])


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    return mask


def inset_quad(quad: np.ndarray, ratio: float) -> np.ndarray:
    center = np.mean(quad, axis=0)
    return center + (quad - center) * (1.0 - ratio)


def enclosing_light_surface(image: np.ndarray, raw: np.ndarray) -> np.ndarray | None:
    """Return the smallest bright connected surface enclosing the RAW text."""
    x, y, width, height = cv2.boundingRect(np.round(raw).astype(np.int32))
    pad_x, pad_y = max(24, width), max(28, height)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2 = min(image.shape[1], x + width + pad_x)
    y2 = min(image.shape[0], y + height + pad_y)
    crop = image[y1:y2, x1:x2]
    if not crop.size:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    bright = np.where(gray >= 185, 255, 0).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = tuple((np.mean(raw, axis=0) - np.array([x1, y1])).astype(float))
    raw_area = max(1.0, cv2.contourArea(cv2.convexHull(raw.astype(np.float32))))
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < raw_area * 1.20 or area > raw_area * 30:
            continue
        if cv2.pointPolygonTest(contour, center, False) < 0:
            continue
        hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
        hull += np.array([x1, y1], dtype=np.float32)
        candidates.append((area, hull))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def classify_surface(image: np.ndarray, quad: np.ndarray, angle: float) -> tuple[str, float, dict]:
    mask = polygon_mask(image.shape[:2], quad)
    pixels = image[mask > 0]
    if not len(pixels):
        return "SURFACE_AMBIGUOUS", 0.0, {}
    gray = np.mean(pixels, axis=1)
    chroma = np.max(pixels, axis=1) - np.min(pixels, axis=1)
    light = float(np.mean(gray >= 210))
    dark = float(np.mean(gray <= 80))
    colorful = float(np.mean(chroma >= 30))
    variance = float(np.var(gray))
    if 2.5 <= abs(angle) <= 28 and light >= 0.45:
        kind, confidence = "TILTED_CARD", min(0.99, 0.72 + light * 0.25)
    elif light >= 0.62 and colorful < 0.18:
        kind, confidence = "BUBBLE_FLAT", min(0.98, 0.70 + light * 0.25)
    elif variance < 900 and colorful >= 0.10:
        kind, confidence = "BUBBLE_COLORED", 0.78
    elif dark >= 0.32 and colorful < 0.25:
        kind, confidence = "CHAT_UI", 0.72
    else:
        kind, confidence = "ARTWORK_TEXT", 0.64
    return kind, round(confidence, 3), {
        "light_ratio": round(light, 3), "dark_ratio": round(dark, 3),
        "color_ratio": round(colorful, 3), "gray_variance": round(variance, 1),
        "median_rgb": np.median(pixels, axis=0).round().astype(int).tolist(),
    }


def analyze(original_path: Path, ocr_path: Path, translations_path: Path, output_path: Path) -> dict:
    started = time.perf_counter()
    with Image.open(original_path) as source:
        image = np.asarray(source.convert("RGB"))
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    translations = json.loads(translations_path.read_text(encoding="utf-8"))
    regions = {item["id"]: item for item in ocr.get("regions", [])}
    page_text=" ".join(str(item.get("text","")).upper() for item in regions.values())
    page_compact=re.sub(r"\s+","",page_text)
    page_width=int(ocr.get("width") or image.shape[1])
    page_height=int(ocr.get("height") or image.shape[0])
    whole_page_source_promo=bool(
        page_height <= page_width*.75
        and "YOU CAN READ" in page_text
        and "CHAPTERS" in page_text
        and re.search(r"MADARA(?:SCAN[S5])?\.?C[O0][MN]",page_compact)
    ) or bool(
        page_width*.9 <= page_height <= page_width*1.8
        and re.search(r"E[VY]ASCAN[S5]",page_compact)
        and "READ" in page_compact and ("ONLYAT" in page_compact or "READONLY" in page_compact)
        and "SUPPORT" in page_compact
        and ("REACHOUT" in page_compact or "DISCORD" in page_compact)
    )
    from branding_v52 import stacked_source_promo_header
    source_header_bottom=stacked_source_promo_header(image,list(regions.values()))
    surfaces = []
    claims: list[tuple[str, np.ndarray]] = []
    for group in translations.get("groups", []):
        source_text = " ".join(str(group.get("text", "")).upper().split())
        group_top=int(group.get("bbox",{}).get("y",image.shape[0]))
        source_credit = whole_page_source_promo or (source_header_bottom and group_top < source_header_bottom) or bool(re.search(
            r"(?:SCAN(?:S|LATION)?|THUNDER|MADARA|EVASCANS?|READ\s+(?:ONLY\s+)?AT|YOU\s+CAN\s+READ|HTTPS?://|WWW\.|\.(?:COM|ORG|NET)\b|"
            r"DISCORD|RECRUIT(?:MENT|ING)?|PROOFREADERS?|TRANSLATORS?|JOIN\b.{0,30}\bDISCORD)",
            source_text,
        ))
        source_regions = [regions[item] for item in group.get("source_region_ids", []) if item in regions]
        angle = source_angle(source_regions)
        raw = order_quad(raw_quad(source_regions, group["bbox"], angle))
        outer = enclosing_light_surface(image, raw)
        polygon = outer if outer is not None else raw
        kind, confidence, stats = classify_surface(image, polygon, angle)
        # Large spiky/shout balloons contain speed lines and thick borders, so
        # their full polygon can fall just below the generic 62% white cutoff.
        # A detected enclosing bright component plus a light median is stronger
        # evidence than the raw ratio alone; keep it on the flat-bubble cleanup
        # path so every OCR line is erased instead of glyph-inpainted partially.
        if (outer is not None and kind == "ARTWORK_TEXT"
                and stats.get("light_ratio", 0) >= .48
                and stats.get("dark_ratio", 1) <= .20
                and stats.get("color_ratio", 1) <= .22
                and float(np.median(stats.get("median_rgb", [0]))) >= 185):
            kind, confidence = "BUBBLE_FLAT", .84
        margin = 0.12 if kind in {"BUBBLE_FLAT", "BUBBLE_COLORED"} else 0.08
        safe = inset_quad(polygon, margin)
        mask = polygon_mask(image.shape[:2], safe)
        collisions = []
        for other_id, other_mask in claims:
            overlap = int(np.count_nonzero((mask > 0) & (other_mask > 0)))
            if overlap > min(np.count_nonzero(mask), np.count_nonzero(other_mask)) * 0.18:
                collisions.append(other_id)
        claims.append((group["id"], mask))
        if collisions:
            confidence = min(confidence, 0.49)
        surfaces.append({
            "id": group["id"], "surface_type": kind, "confidence": confidence,
            "style": group.get("style", "dialogue"),
            "source_equals_translation": " ".join(group.get("text", "").upper().split())
            == " ".join(group.get("translation", "").upper().split()),
            "source_credit": source_credit,
            "angle_degrees": angle, "polygon": polygon.round(2).tolist(),
            "raw_polygon": raw.round(2).tolist(),
            "safe_polygon": safe.round(2).tolist(), "collides_with": collisions,
            "requires_ai_layout": bool(confidence < 0.60), "stats": stats,
            "expanded_from_raw": bool(outer is not None),
        })
    # Collision ownership is symmetric. The streaming detector sees only
    # earlier claims, so mirror every relation after all surfaces are built.
    surfaces_by_id = {item["id"]: item for item in surfaces}
    for item in surfaces:
        for other_id in list(item.get("collides_with", [])):
            other = surfaces_by_id.get(other_id)
            if other is not None and item["id"] not in other["collides_with"]:
                other["collides_with"].append(item["id"])
                other["confidence"] = min(float(other.get("confidence", 0)), 0.49)
                other["requires_ai_layout"] = True
    for left_index, left in enumerate(surfaces):
        left_poly = np.asarray(left.get("polygon") or [], dtype=np.float32)
        if len(left_poly) < 4:
            continue
        lx1, ly1 = np.min(left_poly, axis=0); lx2, ly2 = np.max(left_poly, axis=0)
        if lx2 - lx1 < image.shape[1] * 0.75:
            continue
        for right in surfaces[left_index + 1:]:
            right_poly = np.asarray(right.get("polygon") or [], dtype=np.float32)
            if len(right_poly) < 4:
                continue
            rx1, ry1 = np.min(right_poly, axis=0); rx2, ry2 = np.max(right_poly, axis=0)
            if rx2 - rx1 < image.shape[1] * 0.75:
                continue
            gap = max(0.0, max(ly1, ry1) - min(ly2, ry2))
            if gap <= min(ly2 - ly1, ry2 - ry1) * 0.18:
                for item, other in ((left, right), (right, left)):
                    if other["id"] not in item["collides_with"]:
                        item["collides_with"].append(other["id"])
                    item["confidence"] = min(float(item.get("confidence", 0)), 0.49)
                    item["requires_ai_layout"] = True
    report = {
        "version": 2, "dimensions": [image.shape[1], image.shape[0]],
        "seconds": round(time.perf_counter() - started, 3),
        "surface_count": len(surfaces),
        "ambiguous_count": sum(item["requires_ai_layout"] for item in surfaces),
        "surfaces": surfaces,
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate(original_path: Path, cleaned_path: Path, output_path: Path, mask_path: Path,
             surfaces_path: Path, report_path: Path) -> dict:
    started = time.perf_counter()
    with Image.open(original_path) as source, Image.open(cleaned_path) as cleaned_source, Image.open(output_path) as result:
        original = np.asarray(source.convert("RGB"))
        cleaned = np.asarray(cleaned_source.convert("RGB"))
        output = np.asarray(result.convert("RGB"))
    dimensions_ok = original.shape == cleaned.shape == output.shape
    if not dimensions_ok:
        checks = {"dimensions_ok": False}
    else:
        with Image.open(mask_path) as source:
            # Keep the undilated mask: it marks exactly the RAW glyph pixels the
            # removal stage erased. Residual checks anchor on those pixels
            # instead of guessing which pixels are lettering from brightness.
            removal_glyphs = np.asarray(source.convert("L")) > 0
            removal_allowed = removal_glyphs
        # Inpainting necessarily samples and blends background around the OCR
        # glyph mask. Permit that local context; otherwise a correctly removed
        # long promo line is misclassified as an artwork-sized foreign edit.
        # The allowance remains anchored to detected removal pixels and does
        # not relax checks elsewhere on the page.
        removal_allowed = cv2.dilate(
            removal_allowed.astype(np.uint8),
            np.ones((65, 65), np.uint8),
            iterations=1,
        ) > 0
        surface_data = json.loads(surfaces_path.read_text(encoding="utf-8"))
        typeset_path=report_path.parent/"typeset-report.json"
        intentionally_preserved=set()
        promotion_allowed = np.zeros(original.shape[:2], dtype=np.uint8)
        if typeset_path.exists():
            typeset_data = json.loads(typeset_path.read_text(encoding="utf-8"))
            for result in typeset_data.get("results",[]):
                if (result.get("status")=="PRESERVED"
                    and result.get("reason") in {"UNFIT_SOURCE_RESTORED","SOURCE_PRESERVED"}):
                    intentionally_preserved.add(result.get("id"))
            for promotion in typeset_data.get("automatic_promotions", []):
                if promotion.get("action") != "REMOVED_PROMOTION":
                    continue
                bbox = promotion.get("affected_bbox") or promotion.get("bbox") or {}
                x = max(0, int(bbox.get("x", 0)))
                y = max(0, int(bbox.get("y", 0)))
                right = min(original.shape[1], x + max(0, int(bbox.get("width", 0))))
                bottom = min(original.shape[0], y + max(0, int(bbox.get("height", 0))))
                if right > x and bottom > y:
                    promotion_allowed[y:bottom, x:right] = 255
        surface_allowed = np.zeros(original.shape[:2], dtype=np.uint8)
        residual_candidates = []
        original_gray = np.mean(original, axis=2)
        output_delta = np.max(np.abs(output.astype(np.int16) - original.astype(np.int16)), axis=2)
        for item in surface_data["surfaces"]:
            cv2.fillConvexPoly(surface_allowed, np.round(item["polygon"]).astype(np.int32), 255)
            raw_mask = polygon_mask(original.shape[:2], np.asarray(item.get("raw_polygon") or item["polygon"])) > 0
            median_rgb = item.get("stats", {}).get("median_rgb", [255, 255, 255])
            background_gray = float(np.mean(median_rgb))
            # Foreground must be the RAW lettering itself, not "pixels far from
            # the surface median". On a dark bubble the median sits only ~24 gray
            # levels from black lettering, so the real text falls below the 45
            # threshold and is never measured, while untouched white artwork
            # inside the same polygon sits ~176 away and is measured instead.
            # Reporting untouched artwork as raw residual failed a chapter whose
            # translation was already clean.
            raw_foreground = raw_mask & removal_glyphs
            if int(np.count_nonzero(raw_foreground)) < 12:
                # The removal mask never covered this surface. Fall back to the
                # median-distance test instead of skipping, so a genuine
                # residual still fails rather than passing through an empty
                # measurement.
                raw_foreground = raw_mask & (np.abs(original_gray - background_gray) >= 45)
            if (item.get("source_credit") or item.get("source_equals_translation")
                    or item.get("id") in intentionally_preserved):
                continue
            # Small SFX/hesitation marks may intentionally be preserved when a
            # replacement would be less readable. Their unchanged pixels are
            # therefore not evidence of missed dialogue cleanup.
            if item.get("style") == "sfx":
                continue
            # OCR fragments this small are typically pieces of a logo/SFX and
            # do not provide enough pixels for a reliable residual verdict.
            # Their unchanged ratios previously generated duplicate reviews
            # even when branding fully covered the source mark.
            if int(np.count_nonzero(raw_mask)) < 1200:
                continue
            if np.count_nonzero(raw_foreground) >= 12:
                unchanged_ratio = float(np.mean(output_delta[raw_foreground] <= 6))
                # Replacement glyphs naturally overlap part of the RAW glyph
                # footprint, especially white outlined text over artwork and
                # white text on black captions. Ratios around two thirds are
                # normal overlap; require a clearly dominant unchanged signal
                # before claiming that the English source remains.
                if unchanged_ratio >= 0.90:
                    residual_candidates.append({
                        "id": item["id"], "unchanged_foreground_ratio": round(unchanged_ratio, 3),
                    })
        # Surface polygons originate from tight RAW OCR bounds. Readable
        # Indonesian wrapping and font antialiasing may legitimately extend a
        # small distance into the same bubble/open-caption background. Permit a
        # 24 px gutter (3% of an 800 px page), while still detecting meaningful
        # edits elsewhere in the illustration.
        surface_allowed = cv2.dilate(surface_allowed, np.ones((49, 49), np.uint8), iterations=1)
        changed = np.any(output != original, axis=2)
        allowed = removal_allowed | (surface_allowed > 0) | (promotion_allowed > 0)
        outside = int(np.count_nonzero(changed & ~allowed))
        render_delta = np.any(output != cleaned, axis=2)
        outside_surface = int(np.count_nonzero(render_delta & (surface_allowed == 0)))
        outside_mask = np.where(changed & ~allowed, 255, 0).astype(np.uint8)
        outside_contours, _ = cv2.findContours(outside_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outside_components = []
        for contour in sorted(outside_contours, key=cv2.contourArea, reverse=True)[:12]:
            x, y, width, height = cv2.boundingRect(contour)
            outside_components.append({
                "x": x, "y": y, "width": width, "height": height,
                "area": round(float(cv2.contourArea(contour)), 1),
            })
        checks = {
            "dimensions_ok": True,
            "outside_allowed_pixels": outside,
            "outside_surface_render_pixels": outside_surface,
            "ambiguous_surfaces": int(surface_data.get("ambiguous_count", 0)),
            "outside_components": outside_components,
            "residual_candidates": residual_candidates,
            "intentional_residual_ids": sorted(item for item in intentionally_preserved if item),
        }
    reasons = []
    if not checks.get("dimensions_ok"):
        reasons.append("DIMENSION_MISMATCH")
    # A handful of isolated pixels can be font antialiasing at the edge of the
    # dilated safety mask. Keep reporting their count, but only fail when the
    # spill is large enough to be visually meaningful.
    # Thin connected components are translated glyph rows extending beyond a
    # tight RAW OCR polygon, especially white narration on black panels. The
    # typesetter independently proves every rendered pixel stays inside its
    # selected layout box. QC should catch displaced artwork-sized blobs here,
    # not re-flag valid longer Indonesian letterforms.
    outside_components = checks.get("outside_components", [])
    artwork_sized_spill = any(
        min(int(item.get("width", 0)), int(item.get("height", 0))) >= 48
        for item in outside_components
    )
    # Long pages can contain many perfectly valid thin glyph rows; their total
    # antialiased pixels add up even though no individual component resembles
    # an artwork edit. Component geometry is the stronger signal here.
    if checks.get("outside_allowed_pixels", 0) >= 75000 or artwork_sized_spill:
        reasons.append("PIXEL_OUTSIDE_ALLOWED_MASK")
    if checks.get("residual_candidates"):
        reasons.append("RAW_TEXT_RESIDUAL")
    # Shadow V2 reports render expansion but does not fail production until
    # typesetting consumes the same safe polygons.
    status = "PASS" if not reasons else "FAIL"
    report = {
        "version": 2, "mode": "shadow", "status": status,
        "seconds": round(time.perf_counter() - started, 3),
        "checks": checks, "review_reasons": reasons,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("original", type=Path)
    analyze_parser.add_argument("ocr", type=Path)
    analyze_parser.add_argument("translations", type=Path)
    analyze_parser.add_argument("--output", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("original", type=Path)
    validate_parser.add_argument("cleaned", type=Path)
    validate_parser.add_argument("rendered", type=Path)
    validate_parser.add_argument("mask", type=Path)
    validate_parser.add_argument("surfaces", type=Path)
    validate_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "analyze":
        result = analyze(args.original, args.ocr, args.translations, args.output)
    else:
        result = validate(args.original, args.cleaned, args.rendered, args.mask, args.surfaces, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
