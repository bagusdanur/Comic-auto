#!/usr/bin/env python3
"""Phase 4: auto-fit Indonesian dialogue onto the Phase 3 cleaned image."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from source_text_color import source_text_color


UNICODE_PUNCTUATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u2026": "...", "\u00a0": " ", "\u202f": " ", "\u3000": " ",
})


def sanitize_typeset_text(value: object) -> str:
    """Return font-safe Indonesian text without tofu/missing-glyph boxes.

    Translation providers occasionally return zero-width controls, foreign
    quote marks, emoji, CJK punctuation, or a literal replacement square. The
    production comic fonts do not cover those glyphs, so Pillow renders a box.
    Indonesian output only needs printable ASCII after punctuation and accent
    normalization; unsupported characters become spaces to avoid joining words.
    """
    text = unicodedata.normalize("NFKC", str(value or "")).translate(UNICODE_PUNCTUATION)
    safe: list[str] = []
    for character in text:
        if character in "\r\n\t":
            safe.append(" ")
            continue
        if " " <= character <= "~":
            safe.append(character)
            continue
        decomposed = unicodedata.normalize("NFKD", character)
        ascii_equivalent = "".join(
            item for item in decomposed
            if " " <= item <= "~" and not unicodedata.combining(item)
        )
        safe.append(ascii_equivalent or " ")
    cleaned = " ".join("".join(safe).split())
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str] | None:
    words = text.split()
    if not words:
        return None
    lines: list[str] = []
    current = words[0]
    if font.getlength(current) > max_width:
        return None
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.getlength(candidate) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
            if font.getlength(current) > max_width:
                return None
    lines.append(current)
    return lines


def balanced_wrap_text(text: str, font: ImageFont.FreeTypeFont,
                       max_width: int) -> list[str] | None:
    """Find the narrowest width that preserves the full-width line count."""
    greedy = wrap_text(text, font, max_width)
    if not greedy or len(greedy) <= 1:
        return greedy
    widest_word = math.ceil(max(font.getlength(word) for word in text.split()))
    low, high = min(max_width, widest_word), max_width
    while low < high:
        middle = (low + high) // 2
        candidate = wrap_text(text, font, middle)
        if candidate is not None and len(candidate) <= len(greedy):
            high = middle
        else:
            low = middle + 1
    return wrap_text(text, font, high)


def fit_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: str,
    width: int,
    height: int,
    minimum: int,
    maximum: int,
    spacing_ratio: float,
    stroke_width: int,
    max_lines: int | None = None,
    balanced: bool = False,
) -> dict | None:
    for size in range(maximum, minimum - 1, -1):
        font = ImageFont.truetype(font_path, size=size)
        lines = balanced_wrap_text(text, font, width) if balanced else wrap_text(text, font, width)
        if not lines:
            continue
        if max_lines is not None and len(lines) > max_lines:
            continue
        wrapped = "\n".join(lines)
        spacing = max(1, round(size * spacing_ratio))
        bbox = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, spacing=spacing, align="center",
            stroke_width=stroke_width,
        )
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        if text_width <= width and text_height <= height:
            return {
                "font": font,
                "font_size": size,
                "wrapped": wrapped,
                "spacing": spacing,
                "text_bbox": bbox,
                "line_count": len(lines),
            }
    return None


def is_rectangular_caption(image: np.ndarray, bbox: dict) -> bool:
    """Detect long straight borders around a text group (narration/caption box)."""
    height, width = image.shape[:2]
    pad_x = max(18, round(int(bbox["width"]) * 0.22))
    # Caption boxes often have generous vertical whitespace around a short line
    # of text, so search farther above/below than a speech balloon needs.
    pad_y = max(32, round(int(bbox["height"]) * 2.20))
    x1 = max(0, int(bbox["x"]) - pad_x)
    y1 = max(0, int(bbox["y"]) - pad_y)
    x2 = min(width, int(bbox["x"] + bbox["width"]) + pad_x)
    y2 = min(height, int(bbox["y"] + bbox["height"]) + pad_y)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return False
    gray = np.mean(crop, axis=2)
    dark = gray < 85
    def longest_run(values: np.ndarray) -> int:
        padded = np.pad(values.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        return int(np.max(changes[1::2] - changes[::2], initial=0))

    # Measure continuous straight strokes instead of density across the complete
    # crop. This still recognizes a box when its sides extend well past the OCR
    # bbox, while speech/thought balloon curves do not form four long runs.
    min_horizontal = max(24, round(int(bbox["width"]) * 0.72))
    min_vertical = max(24, round(int(bbox["height"]) * 0.72))
    horizontal_lines = sum(longest_run(row) >= min_horizontal for row in dark)
    vertical_lines = sum(longest_run(column) >= min_vertical for column in dark.T)
    return bool(horizontal_lines >= 2 and vertical_lines >= 2)


def source_rotation(regions: list[dict]) -> float:
    """Baseline angle of the RAW lettering, in degrees.

    Two families are recognised: slightly tilted cards (2.5-24 degrees) and
    captions set at a full right angle. The detector hands a right-angle
    caption over as a tall *axis-aligned* box, so the polygon's top edge is
    horizontal and the slant test below reports 0 -- which is why a vertical
    side caption used to come back as horizontal typeset text.
    """
    bboxes = [region.get("bbox") or {} for region in regions]
    tall = [
        region for region, box in zip(regions, bboxes)
        if int(box.get("height", 0)) >= 1.5 * max(1, int(box.get("width", 0)))
    ]
    if tall and len(tall) * 2 >= len(regions):
        # get_rotate_crop_image deskews a tall crop with np.rot90 (counter-
        # clockwise) and only then lets the textline angle head decide on a
        # further 180 flip. angle==1 therefore means the printed letters run
        # counter-clockwise of upright, so the reader turns the page clockwise
        # to read them and the translation must be painted at -90 to match.
        # Missing angle data (older OCR reports) falls back to that same
        # convention, which is what every scanned side caption has used so far.
        flipped = sum(1 for region in tall if int(region.get("orientation_angle", 1)) == 1)
        return -90.0 if flipped * 2 >= len(tall) else 90.0
    angles = []
    for region in regions:
        polygon = region.get("polygon") or []
        if len(polygon) >= 2:
            dx, dy = polygon[1][0] - polygon[0][0], polygon[1][1] - polygon[0][1]
            if dx:
                angles.append(math.degrees(math.atan2(dy, dx)))
    angle = float(np.median(angles)) if angles else 0.0
    return round(angle, 2) if 2.5 <= abs(angle) <= 24 else 0.0


def raw_oriented_bounds(regions: list[dict], angle: float, fallback: dict) -> dict:
    """Exact oriented bounds of RAW OCR polygons at their measured baseline."""
    points = [point for region in regions for point in (region.get("polygon") or [])]
    if not points:
        return {
            "center": (int(fallback["x"]) + int(fallback["width"]) / 2,
                       int(fallback["y"]) + int(fallback["height"]) / 2),
            "width": int(fallback["width"]), "height": int(fallback["height"]),
        }
    radians = math.radians(angle)
    axis_x = np.array([math.cos(radians), math.sin(radians)], dtype=np.float32)
    axis_y = np.array([-math.sin(radians), math.cos(radians)], dtype=np.float32)
    cloud = np.asarray(points, dtype=np.float32)
    projected_x, projected_y = cloud @ axis_x, cloud @ axis_y
    min_x, max_x = float(projected_x.min()), float(projected_x.max())
    min_y, max_y = float(projected_y.min()), float(projected_y.max())
    center = axis_x * ((min_x + max_x) / 2) + axis_y * ((min_y + max_y) / 2)
    return {
        "center": (float(center[0]), float(center[1])),
        "width": max(1, round(max_x - min_x)),
        "height": max(1, round(max_y - min_y)),
    }


def exact_raw_overlay(draw, translation, font_path, preferred_size, minimum_size,
                      spacing_ratio, source_regions, angle, fallback_bbox, raw_polygon=None):
    box = raw_oriented_bounds(source_regions, angle, fallback_bbox)
    width, height = box["width"], box["height"]
    fitted = fit_text(draw, translation, font_path, max(1, round(width * 0.90)),
                      max(1, round(height * 0.86)), minimum_size, preferred_size,
                      spacing_ratio, 0)
    if not fitted or fitted["font_size"] < min(20, preferred_size):
        # Keep the RAW centre/angle and transparent background, but allow a
        # longer Indonesian label to borrow a little nearby bubble/card space.
        source_length = len(" ".join(str(region.get("text", "")) for region in source_regions))
        length_ratio = len(translation) / max(1, source_length)
        width = round(width * min(1.90, max(1.82, length_ratio ** 0.45)))
        height = round(height * min(1.48, max(1.32, length_ratio ** 0.22)))
        expanded_fit = fit_text(draw, translation, font_path, max(1, round(width * 0.94)),
                          max(1, round(height * 0.90)), minimum_size, preferred_size,
                          spacing_ratio, 0)
        if expanded_fit:
            fitted = expanded_fit
        if not fitted:
            return None
    text_bbox = fitted["text_bbox"]
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.multiline_text(
        ((width - text_width) / 2 - text_bbox[0], (height - text_height) / 2 - text_bbox[1]),
        fitted["wrapped"], font=fitted["font"], fill=(12, 12, 12, 255),
        spacing=fitted["spacing"], align="center", stroke_width=0,
    )
    if angle:
        layer = layer.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=True)
    center_x, center_y = box["center"]
    paste_x, paste_y = round(center_x - layer.width / 2), round(center_y - layer.height / 2)
    polygon = np.asarray(raw_polygon, dtype=np.float32) if raw_polygon else None
    if polygon is None or polygon.size == 0:
        radians = math.radians(angle)
        axis_x = np.array([math.cos(radians), math.sin(radians)])
        axis_y = np.array([-math.sin(radians), math.cos(radians)])
        center = np.array([center_x, center_y])
        polygon = np.asarray([
            center - axis_x * width / 2 - axis_y * height / 2,
            center + axis_x * width / 2 - axis_y * height / 2,
            center + axis_x * width / 2 + axis_y * height / 2,
            center - axis_x * width / 2 + axis_y * height / 2,
        ], dtype=np.float32)
    return layer, fitted, polygon, paste_x, paste_y


def order_quad(points: np.ndarray) -> np.ndarray:
    """Return four points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return ordered


def enclosing_tilted_surface(image: np.ndarray, bbox: dict, rotation: float) -> dict | None:
    """Find the white card containing a tilted OCR group."""
    if not rotation:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bright = np.where(gray >= 205, 255, 0).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bx, by = int(bbox["x"]), int(bbox["y"])
    bw, bh = int(bbox["width"]), int(bbox["height"])
    center = (bx + bw / 2, by + bh / 2)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < bw * bh * 0.72 or area > image.shape[0] * image.shape[1] * 0.22:
            continue
        if cv2.pointPolygonTest(contour, center, False) < 0:
            continue
        rect = cv2.minAreaRect(contour)
        rw, rh = rect[1]
        if min(rw, rh) < 28:
            continue
        fill_ratio = area / max(1.0, rw * rh)
        # Speech cards may have tails, clipped corners, or be partly hidden by
        # another photo; their useful min-area rectangle is still reliable.
        if fill_ratio < 0.45:
            continue
        quad = order_quad(cv2.boxPoints(rect))
        top = np.linalg.norm(quad[1] - quad[0])
        bottom = np.linalg.norm(quad[2] - quad[3])
        left = np.linalg.norm(quad[3] - quad[0])
        right = np.linalg.norm(quad[2] - quad[1])
        width, height = int(round(max(top, bottom))), int(round(max(left, right)))
        if width < bw * 0.60 or height < bh * 0.65:
            continue
        candidates.append((area, quad, width, height, fill_ratio))
    if not candidates:
        return None
    _, quad, width, height, fill_ratio = min(candidates, key=lambda item: item[0])
    if width > bw * 1.80 or height > bh * 2.0:
        # Touching/overlapping white cards can become one connected component.
        # Split that component back into one surface per OCR group, preserving
        # the group's measured baseline angle and adding a modest paper gutter.
        # Do not enlarge an OCR group here: adjacent social cards frequently
        # overlap as paper, while their original lettering areas do not.
        local_width = max(32, round(bw * 1.02))
        local_height = max(28, round(bh * 1.04))
        local_rect = (center, (local_width, local_height), rotation)
        quad = order_quad(cv2.boxPoints(local_rect))
        width, height = local_width, local_height
        fill_ratio = 1.0
    return {"quad": quad, "width": width, "height": height,
            "fill_ratio": round(float(fill_ratio), 3)}


def polygon_brightness(image: np.ndarray, quad: np.ndarray) -> float:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
    pixels = image[mask.astype(bool)]
    return float(np.mean(np.mean(pixels, axis=1) >= 190)) if len(pixels) else 0.0


def enclosing_bubble_box(
    image: np.ndarray,
    bbox: dict,
    contours: list[np.ndarray],
    conservative_inset: bool = True,
) -> dict | None:
    """Expand a tight OCR group into the safe interior of its enclosing balloon."""
    bx, by = int(bbox["x"]), int(bbox["y"])
    bw, bh = int(bbox["width"]), int(bbox["height"])
    center = (bx + bw / 2, by + bh / 2)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width * height < bw * bh * 2.2 or width > image.shape[1] * 0.95:
            continue
        if not (x <= center[0] <= x + width and y <= center[1] <= y + height):
            continue
        if cv2.pointPolygonTest(contour, center, False) < 0:
            continue
        candidates.append((width * height, x, y, width, height))
    if not candidates:
        return None
    _, x, y, width, height = min(candidates)
    # A conservative inset approximates the largest readable rectangle inside
    # round/oval bubbles without letting text touch their outline.
    if conservative_inset:
        inset_x, inset_y = round(width * 0.16), round(height * 0.17)
    else:
        # V4 calculates dynamic ellipse budgets; preserve full balloon measure
        inset_x, inset_y = round(width * 0.04), round(height * 0.04)
    candidate = {
        "x": max(0, x + inset_x), "y": max(0, y + inset_y),
        "width": max(1, width - inset_x * 2), "height": max(1, height - inset_y * 2),
    }
    if candidate["width"] < bw * 0.70 or candidate["height"] < bh * 1.30:
        return None
    # The detected interior may be narrower than the source text because round
    # sides taper. Preserve the already-known-safe OCR rectangle and only add
    # the extra whitespace discovered around it.
    right = max(candidate["x"] + candidate["width"], bx + bw)
    bottom = max(candidate["y"] + candidate["height"], by + bh)
    candidate["x"], candidate["y"] = min(candidate["x"], bx), min(candidate["y"], by)
    candidate["width"], candidate["height"] = right - candidate["x"], bottom - candidate["y"]
    return candidate


def crosses_stacked_card_boundary(
    raw_bbox: dict, expanded_bbox: dict, has_straight_card_border: bool = False
) -> bool:
    """Reject a connected-component expansion that joined stacked paper cards.

    OCR boxes for caption cards are commonly wide and shallow.  If two white
    cards touch, contour detection sees one tall white component and mistakes
    it for a speech balloon.  Expanding into that component lets translated
    text cross the original card edge.  Real dialogue balloons tend to start
    with less extreme OCR geometry, so keep this deliberately narrow.
    """
    raw_w = max(1, int(raw_bbox["width"]))
    raw_h = max(1, int(raw_bbox["height"]))
    expanded_w = max(1, int(expanded_bbox["width"]))
    expanded_h = max(1, int(expanded_bbox["height"]))
    raw_area = raw_w * raw_h
    expanded_area = expanded_w * expanded_h
    return (
        has_straight_card_border
        and raw_w / raw_h >= 2.15
        and expanded_h / raw_h >= 2.20
        and expanded_area / raw_area >= 2.60
    )


def safe_surface_bbox(surface: dict | None, raw_bbox: dict) -> dict | None:
    """Return the V2 safe interior when contour-only bubble detection misses it."""
    if not surface or surface.get("surface_type") != "BUBBLE_FLAT":
        return None
    if float(surface.get("confidence", 0)) < 0.85 or surface.get("collides_with"):
        return None
    polygon = surface.get("safe_polygon") or []
    if len(polygon) < 4:
        return None
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x1, y1 = math.floor(min(xs)), math.floor(min(ys))
    x2, y2 = math.ceil(max(xs)), math.ceil(max(ys))
    candidate = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
    raw_area = max(1, int(raw_bbox["width"]) * int(raw_bbox["height"]))
    if candidate["width"] * candidate["height"] < raw_area * 1.35:
        return None
    return candidate


def prefer_v2_over_contour(contour: dict | None, panel_edge_balloon: bool,
                           straight_border: bool) -> bool:
    if contour is None:
        return not straight_border or panel_edge_balloon
    return bool(
        panel_edge_balloon
        and contour["height"] > contour["width"] * 1.05
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cleaned_image", type=Path)
    parser.add_argument("translations_json", type=Path)
    parser.add_argument("--ocr-json", type=Path)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--removal-report", type=Path)
    parser.add_argument("--surface-map", type=Path)
    parser.add_argument("--hard-clip", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/typeset.png"))
    parser.add_argument("--report", type=Path, default=Path("output/typeset-report.json"))
    parser.add_argument(
        "--font",
        default=os.getenv("TYPESET_FONT", "assets/fonts/wildwords/CC Wild Words Roman.ttf"),
    )
    parser.add_argument("--font-dialogue", default="assets/fonts/wildwords/CC Wild Words Roman.ttf")
    parser.add_argument("--font-narration", default="assets/fonts/wildwords/CC Wild Words Roman.ttf")
    parser.add_argument("--font-thought", default="assets/fonts/wildwords/CC Wild Words Italic.ttf")
    parser.add_argument("--font-shout", default="assets/fonts/wildwords/CC Wild Words Bold Italic.ttf")
    parser.add_argument("--font-sfx", default="assets/fonts/wildwords/CC Wild Words Bold Italic.ttf")
    parser.add_argument("--font-chat", default="assets/fonts/wildwords/CC Wild Words Roman.ttf")
    parser.add_argument("--min-font-size", type=int, default=int(os.getenv("TYPESET_MIN_FONT_SIZE", "18")))
    parser.add_argument("--max-font-size", type=int, default=int(os.getenv("TYPESET_MAX_FONT_SIZE", "64")))
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--line-spacing", type=float, default=0.12)
    parser.add_argument("--min-scale-ratio", type=float, default=0.52)
    parser.add_argument(
        "--layout-engine", choices=("legacy", "balanced", "v3", "v4", "v5"),
        default=os.getenv("TYPESET_LAYOUT_ENGINE", "legacy").casefold(),
    )
    args = parser.parse_args()
    if args.layout_engine == "v5":
        from typeset_engine_v5 import render_page
        return render_page(args)
    started = time.perf_counter()

    with Image.open(args.cleaned_image) as source:
        original = np.asarray(source.convert("RGB")).copy()
    reference = original
    if args.reference_image:
        with Image.open(args.reference_image) as source:
            reference = np.asarray(source.convert("RGB")).copy()
        if reference.shape != original.shape:
            raise RuntimeError("Reference image dimensions differ from cleaned input")
    rendered = Image.fromarray(original.copy())
    draw = ImageDraw.Draw(rendered)
    reference_gray = np.mean(reference, axis=2).astype(np.uint8)
    outline_mask = (reference_gray < 105).astype(np.uint8) * 255
    outline_mask = cv2.morphologyEx(outline_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    bubble_contours, _ = cv2.findContours(outline_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    groups = json.loads(args.translations_json.read_text(encoding="utf-8"))["groups"]
    original_group_boxes = [item["bbox"] for item in groups]
    cleaned_ids = None
    if args.removal_report:
        removal = json.loads(args.removal_report.read_text(encoding="utf-8"))
        cleaned_ids = {item["id"] for item in removal["results"] if item["status"] in {"CLEANED", "INPAINTED"}}
        preserved_ids = {item["id"] for item in removal["results"] if item["status"] == "PRESERVED"}
        removal_status = {item["id"]: item["status"] for item in removal["results"]}
        removal_details = {item["id"]: item for item in removal["results"]}
    else:
        preserved_ids = set()
        removal_status = {}
        removal_details = {}
    regions_by_id = {}
    if args.ocr_json:
        ocr = json.loads(args.ocr_json.read_text(encoding="utf-8"))
        regions_by_id = {region["id"]: region for region in ocr["regions"]}
    surface_by_id = {}
    if args.surface_map and args.surface_map.exists():
        surface_data = json.loads(args.surface_map.read_text(encoding="utf-8"))
        surface_by_id = {item["id"]: item for item in surface_data.get("surfaces", [])}
        # Older V2 maps recorded overlap only on the later surface. Make the
        # relation symmetric so neither of two touching balloons can claim the
        # merged white component.
        for item in list(surface_by_id.values()):
            for other_id in list(item.get("collides_with", [])):
                other = surface_by_id.get(other_id)
                if other is not None and item["id"] not in other.setdefault("collides_with", []):
                    other["collides_with"].append(item["id"])
                    other["confidence"] = min(float(other.get("confidence", 0)), 0.49)
                    other["requires_ai_layout"] = True
        surface_items = list(surface_by_id.values())
        for left_index, left in enumerate(surface_items):
            left_poly = np.asarray(left.get("polygon") or [], dtype=np.float32)
            if len(left_poly) < 4:
                continue
            lx1, ly1 = np.min(left_poly, axis=0); lx2, ly2 = np.max(left_poly, axis=0)
            if lx2 - lx1 < reference.shape[1] * 0.75:
                continue
            for right in surface_items[left_index + 1:]:
                right_poly = np.asarray(right.get("polygon") or [], dtype=np.float32)
                if len(right_poly) < 4:
                    continue
                rx1, ry1 = np.min(right_poly, axis=0); rx2, ry2 = np.max(right_poly, axis=0)
                if rx2 - rx1 < reference.shape[1] * 0.75:
                    continue
                vertical_gap = max(0.0, max(ly1, ry1) - min(ly2, ry2))
                if vertical_gap <= min(ly2 - ly1, ry2 - ry1) * 0.18:
                    for item, other in ((left, right), (right, left)):
                        if other["id"] not in item.setdefault("collides_with", []):
                            item["collides_with"].append(other["id"])
                        item["confidence"] = min(float(item.get("confidence", 0)), 0.49)
                        item["requires_ai_layout"] = True
    allowed = np.zeros(original.shape[:2], dtype=bool)
    occupied_text = np.zeros(original.shape[:2], dtype=bool)
    pending_tilted = []
    results = []

    for group_index, group in enumerate(groups):
        bbox = group["bbox"]
        source_regions = [regions_by_id[item] for item in group.get("source_region_ids", []) if item in regions_by_id]
        source_text = " ".join(str(group.get("text", "")).split())
        translated_text = sanitize_typeset_text(group.get("translation", ""))
        source_is_unchanged = source_text.casefold() == translated_text.casefold()
        source_is_scanlator_ad = bool(re.search(
            r"(?:\b(?:we\s+are|we're)\s+recruiting\b|\brecruit(?:ment|ing)?\b|"
            r"\bproofreaders?\b|\b(?:korean|japanese)\s+translators?\b|"
            r"\bjoin\b.{0,30}\bdiscord\b|\bdiscord(?:\.gg|\.cc)?\b|"
            r"\boi\s*scans\b)",
            source_text, re.IGNORECASE,
        ))
        # A source credit, logo, SFX, or scanlator recruitment card is not comic
        # dialogue. Older cleanup reports may have removed it before this rule
        # existed, so restore the exact RAW pixels here and keep it out of QC.
        if source_is_unchanged or source_is_scanlator_ad:
            for item in source_regions:
                line = item["bbox"]
                rx, ry = int(line["x"]), int(line["y"])
                rw, rh = int(line["width"]), int(line["height"])
                rendered.paste(Image.fromarray(reference[ry:ry + rh, rx:rx + rw]), (rx, ry))
                allowed[ry:ry + rh, rx:rx + rw] = True
            results.append({
                "id": group["id"], "status": "PRESERVED",
                "reason": "SOURCE_EQUALS_TRANSLATION" if source_is_unchanged else "SCANLATOR_PROMO",
                "bbox": bbox, "style": group.get("style", "sfx"),
            })
            continue
        if group["id"] in preserved_ids:
            results.append({
                "id": group["id"], "status": "PRESERVED",
                "reason": removal_details.get(group["id"], {}).get("reason", "SOURCE_PRESERVED"),
                "bbox": bbox, "style": group.get("style", "sfx"),
            })
            continue
        if cleaned_ids is not None and group["id"] not in cleaned_ids:
            # Tiny standalone SFX on detailed artwork are safer left intact than
            # destructively inpainted; they must not block an otherwise ready chapter.
            if group.get("style") == "sfx" or (int(bbox["width"]) <= 120 and int(bbox["height"]) <= 55 and len(source_regions) <= 2):
                results.append({
                    "id": group["id"], "status": "PRESERVED", "reason": "SMALL_SFX_ARTWORK",
                    "bbox": bbox, "style": "sfx",
                })
                continue
            results.append({
                "id": group["id"], "status": "NEEDS_REVIEW",
                "reason": "REMOVAL_NOT_SAFE", "bbox": bbox,
            })
            continue
        expanded_bbox = enclosing_bubble_box(
            reference, bbox, bubble_contours,
            conservative_inset=(args.layout_engine != "v4")
        )
        expansion_source = "contour" if expanded_bbox else None
        contour_aspect = (
            int(expanded_bbox["width"]) / max(1, int(expanded_bbox["height"]))
            if expanded_bbox else 0.0
        )
        stacked_card_expansion_rejected = bool(
            expanded_bbox
            # A localized round/oval contour is a real speech balloon. Nearby
            # panel/card strokes may intersect its bbox, but must never cause
            # us to discard it in favour of a full-width white panel surface.
            and not (0.62 <= contour_aspect <= 1.62)
            and crosses_stacked_card_boundary(
                bbox, expanded_bbox, is_rectangular_caption(reference, bbox)
            )
        )
        if stacked_card_expansion_rejected:
            expanded_bbox = None
            expansion_source = None
        # Balloons touching a panel edge are intentionally excluded by the
        # contour detector. Quality V2 already measures their safe interior;
        # use it unless the RAW region has straight caption-card borders.
        v2_surface = surface_by_id.get(group["id"])
        v2_candidate = safe_surface_bbox(v2_surface, bbox)
        straight_border = is_rectangular_caption(reference, bbox)
        if (
            expanded_bbox
            and group.get("style") == "narration"
            and v2_surface
            and v2_surface.get("surface_type") in {"CHAT_UI", "ARTWORK_TEXT"}
        ):
            # Rectangular panel/card strokes can form a false closed contour.
            # Narration must stay on its verified RAW/V2 surface, not expand
            # across neighbouring black boxes as if the panel were a balloon.
            expanded_bbox = None
            expansion_source = None
        if v2_candidate:
            # A panel-edge balloon can look rectangular because only its two
            # curved side arcs are visible in the crop. V2 identifies these as
            # near-full-width, uniformly white surfaces. That signature is
            # unlike stacked cards, whose merged polygon includes black gaps.
            panel_edge_balloon = bool(
                v2_candidate
                and v2_candidate["width"] >= reference.shape[1] * 0.78
                and float((v2_surface.get("stats") or {}).get("light_ratio", 0)) >= 0.88
            )
            clean_expanded_caption = bool(
                not (v2_surface.get("collides_with") or [])
                and float(v2_surface.get("confidence", 0)) >= 0.87
                and float((v2_surface.get("stats") or {}).get("light_ratio", 0)) >= 0.87
                and v2_candidate["height"] >= int(group["bbox"]["height"]) * 1.65
            )
            v2_is_wide_strip = bool(
                v2_candidate["width"] >= int(group["bbox"]["width"]) * 1.8
                and v2_candidate["height"] <= v2_candidate["width"] * 0.80
            )
            v2_is_merged_panel = bool(
                v2_candidate["width"] >= reference.shape[1] * 0.85
                and int(group["bbox"]["width"]) <= reference.shape[1] * 0.70
                and v2_candidate["width"] >= int(group["bbox"]["width"]) * 1.65
            )
            # Prefer the contour whenever it describes a plausible round/wide
            # balloon. V2 can merge an entire white panel and would then place
            # text across a smaller bubble inside it. Override only when no
            # contour exists or the contour became an obviously tall artifact.
            use_v2 = prefer_v2_over_contour(expanded_bbox, panel_edge_balloon, straight_border)
            if not expanded_bbox and clean_expanded_caption:
                use_v2 = True
            v3_trusted_bubble = bool(
                args.layout_engine in {"v3", "v4"}
                and v2_surface.get("surface_type") == "BUBBLE_FLAT"
                and float(v2_surface.get("confidence", 0)) >= 0.84
                and not (v2_surface.get("collides_with") or [])
                and float((v2_surface.get("stats") or {}).get("light_ratio", 0)) >= 0.82
            )
            if v3_trusted_bubble and not expanded_bbox:
                use_v2 = True
            if v2_is_wide_strip and group.get("style", "dialogue") in {"dialogue", "thought", "shout"}:
                use_v2 = False
            if v2_is_merged_panel and group.get("style", "dialogue") in {"dialogue", "thought", "shout"}:
                use_v2 = False
            if use_v2:
                expanded_bbox = v2_candidate
                expansion_source = "quality_v2"
        if expanded_bbox:
            if (
                args.layout_engine in {"v3", "v4"}
                and int(expanded_bbox["width"]) < int(group["bbox"]["width"]) * 0.72
            ):
                # A partial contour (usually one curved shoulder/tail) is not
                # the writable balloon. RAW OCR bounds are safer and wider.
                expanded_bbox = None
                expansion_source = None
        if expanded_bbox:
            ex1, ey1 = expanded_bbox["x"], expanded_bbox["y"]
            ex2, ey2 = ex1 + expanded_bbox["width"], ey1 + expanded_bbox["height"]
            raw_center_x = int(group["bbox"]["x"]) + int(group["bbox"]["width"]) / 2
            raw_center_y = int(group["bbox"]["y"]) + int(group["bbox"]["height"]) / 2
            centers_in_same_contour = []
            for other_index, other in enumerate(original_group_boxes):
                if other_index == group_index:
                    continue
                other_center_x = int(other["x"]) + int(other["width"]) / 2
                other_center_y = int(other["y"]) + int(other["height"]) / 2
                if ex1 <= other_center_x <= ex2 and ey1 <= other_center_y <= ey2:
                    centers_in_same_contour.append((other_center_x, other_center_y))
            # A connected/figure-eight balloon has one outer contour but two
            # independent dialogue centres. Fit each dialogue into the lobe
            # around its own RAW centre, never across the narrow connector.
            if centers_in_same_contour and 0.62 <= expanded_bbox["width"] / max(1, expanded_bbox["height"]) <= 1.62:
                lobe_width = max(int(group["bbox"]["width"]) + 24, round(expanded_bbox["width"] * 0.68))
                lobe_height = max(int(group["bbox"]["height"]) + 24, round(expanded_bbox["height"] * 0.58))
                lobe_width = min(lobe_width, expanded_bbox["width"])
                lobe_height = min(lobe_height, expanded_bbox["height"])
                lobe_x = round(raw_center_x - lobe_width / 2)
                lobe_y = round(raw_center_y - lobe_height / 2)
                lobe_x = min(max(ex1, lobe_x), ex2 - lobe_width)
                lobe_y = min(max(ey1, lobe_y), ey2 - lobe_height)
                expanded_bbox = {
                    "x": lobe_x, "y": lobe_y,
                    "width": lobe_width, "height": lobe_height,
                }
                expansion_source = "contour_lobe"
                ex1, ey1 = lobe_x, lobe_y
                ex2, ey2 = lobe_x + lobe_width, lobe_y + lobe_height
            for other_index, other in enumerate(original_group_boxes):
                if other_index == group_index:
                    continue
                ox1, oy1 = int(other["x"]), int(other["y"])
                ox2, oy2 = ox1 + int(other["width"]), oy1 + int(other["height"])
                overlap = max(0, min(ex2, ox2) - max(ex1, ox1)) * max(0, min(ey2, oy2) - max(ey1, oy1))
                if overlap > int(other["width"]) * int(other["height"]) * 0.12:
                    expanded_bbox = None
                    break
        if expanded_bbox and expansion_source in {"contour", "contour_lobe"}:
            body_aspect = int(expanded_bbox["width"]) / max(1, int(expanded_bbox["height"]))
            speech_style = group.get("style", "dialogue") in {"dialogue", "thought", "shout"}
            if speech_style and body_aspect < 0.62:
                # A long downward tail can make a round balloon look like a
                # very tall contour. Keep only the near-square body around the
                # RAW dialogue centre; the tail is never writable space.
                old = dict(expanded_bbox)
                body_height = min(int(old["height"]), round(int(old["width"]) * 1.24))
                center_y = int(group["bbox"]["y"]) + int(group["bbox"]["height"]) / 2
                body_y = round(center_y - body_height / 2)
                body_y = min(max(int(old["y"]), body_y), int(old["y"] + old["height"] - body_height))
                expanded_bbox = {
                    "x": int(old["x"]), "y": body_y,
                    "width": int(old["width"]), "height": body_height,
                }
                expansion_source = "contour_body"
            elif speech_style and body_aspect > 1.62:
                # Same correction for a long sideways tail.
                old = dict(expanded_bbox)
                body_width = min(int(old["width"]), round(int(old["height"]) * 1.24))
                center_x = int(group["bbox"]["x"]) + int(group["bbox"]["width"]) / 2
                body_x = round(center_x - body_width / 2)
                body_x = min(max(int(old["x"]), body_x), int(old["x"] + old["width"] - body_width))
                expanded_bbox = {
                    "x": body_x, "y": int(old["y"]),
                    "width": body_width, "height": int(old["height"]),
                }
                expansion_source = "contour_body"
        if (
            expanded_bbox
            and args.layout_engine in {"v3", "v4"}
            and int(expanded_bbox["width"]) < int(group["bbox"]["width"]) * 0.72
        ):
            expanded_bbox = None
            expansion_source = None
        if expanded_bbox:
            bbox = expanded_bbox
        layout_stacked_card = bool(
            expanded_bbox
            and v2_surface
            and (v2_surface.get("collides_with") or [])
            and is_rectangular_caption(reference, bbox)
        )
        pad_x = max(3, round(int(bbox["width"]) * 0.04))
        pad_y = max(4, round(int(bbox["height"]) * 0.10))
        # Long narration placed directly on open artwork often has only a tight
        # one-line OCR box even though the surrounding panel has ample room.
        # Give it vertical breathing room so Indonesian can wrap into two
        # readable lines instead of shrinking across one extremely long line.
        if group.get("style") == "narration" and removal_status.get(group["id"]) == "INPAINTED":
            pad_y = max(pad_y, round(int(bbox["height"]) * 0.45))
        elif (removal_status.get(group["id"]) == "INPAINTED"
              and int(bbox["height"]) <= 90
              and int(bbox["width"]) >= int(bbox["height"]) * 2.4):
            pad_y = max(pad_y, round(int(bbox["height"]) * 0.55))
        # Very short lettering placed directly on artwork has a tight OCR box
        # around the glyphs, not a physical bubble boundary. Indonesian often
        # needs a few more pixels than the English word. Add a small transparent
        # layout gutter; this draws no card and changes no surrounding pixels.
        compact_open_text = bool(
            removal_status.get(group["id"]) == "INPAINTED"
            and int(bbox["height"]) <= 55
            and len(str(group.get("translation", ""))) <= 24
            and len(str(group.get("translation", "")).split()) <= 5
        )
        if compact_open_text:
            pad_x = max(pad_x, round(int(bbox["width"]) * 0.20))
            pad_y = max(pad_y, round(int(bbox["height"]) * 0.24))
        x = max(0, int(bbox["x"]) - pad_x + args.margin)
        y = max(0, int(bbox["y"]) - pad_y + args.margin)
        width = min(original.shape[1] - x, int(bbox["width"]) + pad_x * 2 - args.margin * 2)
        height = min(original.shape[0] - y, int(bbox["height"]) + pad_y * 2 - args.margin * 2)
        # Leave an inner gutter around every fitted block. OCR bounds follow
        # glyphs rather than the true balloon, so this keeps italic overhangs
        # and optimistic font metrics away from the bubble outline.
        # A contour bbox is the outer rectangle around a curved balloon.  Its
        # corners are not writable space: fitting against that full rectangle
        # made 42 px dialogue cross circular/oval outlines even though Pillow
        # reported a valid rectangular fit. Reserve the curved shoulders here;
        # straight caption cards keep the smaller conventional gutter.
        expanded_aspect = (
            int(expanded_bbox["width"]) / max(1, int(expanded_bbox["height"]))
            if expanded_bbox else 0.0
        )
        # Connected/figure-eight balloons contain long straight-ish connector
        # strokes that can fool the caption-border detector. A near-square
        # contour is still a balloon and needs curved shoulder margins.
        curved_balloon = bool(
            expanded_bbox
            and (
                not straight_border
                or 0.62 <= expanded_aspect <= 1.62
                or (
                    expansion_source in {"contour", "contour_lobe", "contour_body"}
                    and group.get("style", "dialogue") in {"dialogue", "thought", "shout"}
                )
            )
        )
        # Preserve useful bubble space and RAW-like font scale. Placement is
        # anchored to the RAW OCR centre below; this moderate inset is only a
        # boundary guard, not a reason to make every translation tiny.
        # Give Indonesian a slightly wider measure so sentences wrap naturally
        # instead of stacking one short word per line. RAW anchoring and the
        # bubble boundary still constrain the final placement.
        longest_word = max((len(word) for word in translated_text.split()), default=0)
        word_width_constrained = bool(longest_word >= 10 and len(translated_text.split()) <= 4)
        curved_inset = (
            0.055 if (curved_balloon and word_width_constrained) or args.layout_engine == "v4"
            else 0.22 if curved_balloon and not (0.62 <= expanded_aspect <= 1.62)
            else 0.20
        )
        safe_x = max(3, round(width * (0.05 if layout_stacked_card else curved_inset if curved_balloon else 0.055)))
        safe_y = max(3, round(height * (0.06 if layout_stacked_card else 0.055 if args.layout_engine == "v4" else 0.20 if curved_balloon else 0.065)))
        x += safe_x
        y += safe_y
        width = max(1, width - safe_x * 2)
        height = max(1, height - safe_y * 2)
        rotation = source_rotation(source_regions)
        # The OCR polygons already provide the RAW baseline angle. Keep the
        # original lettering box and rotate only the replacement glyph layer;
        # never warp, enlarge, or reconstruct the surrounding comic surface.
        tilted_surface = None
        if tilted_surface:
            surface_margin_x = max(6, round(tilted_surface["width"] * 0.09))
            surface_margin_y = max(5, round(tilted_surface["height"] * 0.10))
            width = max(1, tilted_surface["width"] - surface_margin_x * 2)
            height = max(1, tilted_surface["height"] - surface_margin_y * 2)
        source_heights = []
        for item in source_regions:
            polygon = item.get("polygon") or []
            if len(polygon) >= 4:
                edge_height = math.hypot(polygon[3][0] - polygon[0][0], polygon[3][1] - polygon[0][1])
                source_heights.append(max(1, round(edge_height)))
            else:
                source_heights.append(int(item["bbox"]["height"]))
        source_line_height = float(np.median(source_heights)) if source_heights else height / 3
        preferred_size = min(args.max_font_size, max(args.min_font_size, round(source_line_height * 1.2)))
        # Dense monologues inside large shout/spiky balloons can technically fit
        # their rectangular OCR bounds while still looking crowded against the
        # irregular shoulders. Match the RAW face scale conservatively instead
        # of maximizing a 9+ line paragraph to the enclosing rectangle.
        if len(translated_text) >= 220 and int(bbox["height"]) >= 260:
            preferred_size = min(preferred_size, max(18, round(source_line_height * 0.82)))
        stacked_text_card = layout_stacked_card
        if stacked_text_card:
            # Straight overlapping narration cards use the RAW face scale;
            # maximizing every short sentence to 42 px makes it cross the card
            # border even when its rectangular bbox technically fits.
            preferred_size = min(preferred_size, max(16, round(source_line_height * 0.78)))
        if rotation:
            # Match the RAW lettering scale. Tilted text expands after rotation,
            # so maximizing it to the surrounding card is visibly wrong.
            preferred_size = min(preferred_size, max(16, round(source_line_height * 0.92)), 24)

        raw_color = source_text_color(reference, original, group["bbox"])
        text_color = tuple(int(value) for value in raw_color.get("color", (15, 15, 15)))
        context_pad_x = max(6, round(int(bbox["width"]) * 0.12))
        context_pad_y = max(8, round(int(bbox["height"]) * 0.30))
        cx1 = max(0, int(bbox["x"]) - context_pad_x)
        cy1 = max(0, int(bbox["y"]) - context_pad_y)
        cx2 = min(reference.shape[1], int(bbox["x"] + bbox["width"]) + context_pad_x)
        cy2 = min(reference.shape[0], int(bbox["y"] + bbox["height"]) + context_pad_y)
        # Determine contrast from the actual cleaned surface, not from the raw
        # reference. A colored source credit may have been safely cleaned to
        # white; sampling the raw would incorrectly choose white text on white.
        context = original[cy1:cy2, cx1:cx2]
        context_gray = np.mean(context, axis=2) if context.size else np.empty(0)
        context_chroma = (np.max(context, axis=2) - np.min(context, axis=2)) if context.size else np.empty(0)
        flat_white_background = bool(
            context.size and np.mean(context_gray >= 225) >= 0.58
            and np.median(context_chroma) <= 12
        )
        # Use a real outline, not a heavy artificial shadow. RAW comic captions
        # on artwork usually use dark fill plus a thin white keyline.
        artwork_background = removal_status.get(group["id"]) == "INPAINTED" and not flat_white_background
        stroke_width = 1 if artwork_background else 0
        stroke_fill = (255, 255, 255) if sum(text_color) <= 420 else (10, 10, 10)
        if artwork_background:
            context_dark = bool(context.size and float(np.median(context_gray)) < 70)
            if context_dark and sum(text_color) <= 420:
                text_color = (248, 248, 248)
                stroke_fill = (10, 10, 10)
        if rotation:
            # Tilted social-card lettering is already positioned by its RAW OCR
            # polygons. Preserve its dark native colour and never add an
            # artificial comic outline merely because black collage surrounds it.
            text_color = tuple(min(55, int(value)) for value in text_color)
            stroke_width = 0
            stroke_fill = text_color
            artwork_background = False
        if tilted_surface:
            # Judge card colour inside its polygon. A rectangular context crop
            # includes the surrounding black collage and used to misclassify a
            # white paper card as artwork, producing noisy outlined lettering.
            surface_mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(surface_mask, np.round(tilted_surface["quad"]).astype(np.int32), 255)
            surface_pixels = original[surface_mask.astype(bool)]
            if len(surface_pixels) and float(np.mean(np.mean(surface_pixels, axis=1) >= 205)) >= 0.45:
                text_color = (12, 12, 12)
                stroke_width = 0
                stroke_fill = text_color
                artwork_background = False
        translation = sanitize_typeset_text(group.get("translation", "")).upper()
        style = group.get("style", "dialogue")
        if style not in {"dialogue", "narration", "thought", "shout", "sfx", "chat"}:
            style = "dialogue"
        rectangular_caption = bool(
            flat_white_background and (straight_border or stacked_text_card) and not curved_balloon
        )
        if rectangular_caption:
            style = "narration"
        colored_chat = bool(context.size and np.mean(context_chroma > 35) >= 0.28 and int(bbox["width"]) < original.shape[1] * 0.62)
        if colored_chat and style in {"dialogue", "thought"}:
            style = "chat"
        mask_ratios = [
            float(item.get("mask_ratio", 0))
            for item in removal_details.get(group["id"], {}).get("regions", [])
        ]
        dense_artwork = bool(
            artwork_background and mask_ratios and max(mask_ratios) >= 0.45
            and style in {"narration", "chat"}
        )
        # Dense outlined narration used to restore the RAW crop here. That made
        # the safety fallback indistinguishable from a successful translation:
        # English remained in the exported chapter. The cleanup stage has
        # already produced a pixel-confined inpaint for this exact glyph mask,
        # so always typeset the translation over that cleaned surface. QC still
        # catches residual glyphs or spill without silently reverting to RAW.
        # Dense outlined promotional copy over artwork cannot be reconstructed
        # faithfully by inpainting: its mask consumes much of the illustration.
        # Cover that source caption with a deliberate dark card so no English
        # outline/noise remains and the surrounding artwork stays untouched.
        # Keep promotional lettering integrated with the illustration, matching
        # the RAW layout. Dense masks are reconstructed by the cleanup stage;
        # the replacement uses a contrasting outline without an opaque card.
        promo_card = False
        selected_font = {
            "dialogue": args.font_dialogue or args.font,
            "narration": args.font_narration,
            "thought": args.font_thought,
            "shout": args.font_shout,
            "sfx": args.font_sfx,
            "chat": args.font_chat,
        }[style]
        style_spacing = {
            "dialogue": max(args.line_spacing, 0.14),
            "narration": max(args.line_spacing, 0.20),
            "thought": max(args.line_spacing, 0.18),
            "shout": max(args.line_spacing, 0.16),
            "sfx": max(args.line_spacing, 0.14),
            "chat": max(args.line_spacing, 0.16),
        }[style]
        raw_box = group["bbox"]
        compact_raw_text = bool(
            int(raw_box["height"]) <= 55
            and len(translation) <= 24
            and len(translation.split()) <= 5
        )
        # A 22 px global floor is appropriate for normal dialogue, but short
        # RAW labels often use an 18-21 px face in a shallow box. Matching that
        # native scale is preferable to erasing the source and exporting an
        # empty bubble.
        # Bubble boundaries are absolute; readability must never be achieved by
        # drawing across them. Allow a curved balloon to shrink below the normal
        # 18 px floor when its translated sentence needs more lines.
        translation_expanded = len(translation) > max(20, round(len(group.get("text", "")) * 1.08))
        shallow_text_box = int(raw_box["width"]) >= int(raw_box["height"]) * 2.5
        layout_constrained = curved_balloon or translation_expanded or shallow_text_box
        fit_minimum = (
            args.min_font_size if args.layout_engine in {"v3", "v4"} and style in {"dialogue", "thought", "shout", "chat"}
            else max(14, args.min_font_size - 8) if layout_constrained
            else max(16, args.min_font_size - 6) if compact_raw_text
            else args.min_font_size
        )
        balanced_layout = bool(
            args.layout_engine in {"balanced", "v3"}
            and style in {"dialogue", "thought", "shout", "chat"}
        )
        fitted = None
        if args.layout_engine == "v4" and style in {"dialogue", "thought", "shout", "chat"}:
            from typeset_engine_v4 import fit_text_v4
            fitted = fit_text_v4(
                draw, translation, selected_font, width, height,
                fit_minimum, preferred_size, style_spacing, stroke_width,
                is_curved_balloon=curved_balloon,
            )
        if not fitted:
            fitted = fit_text(
                draw, translation, selected_font, width, height,
                fit_minimum, preferred_size, style_spacing, stroke_width,
                # Bubble height, not the number of fragmented RAW OCR regions,
                # determines wrapping. A strict line-count cap forced long
                # Indonesian dialogue into tiny fonts even in large empty bubbles.
                # Dark/open-art captions still follow the RAW line budget so they
                # cannot expand into neighbouring panel/card geometry.
                # The OCR line count describes English, not the available surface.
                # Indonesian routinely needs extra wraps. Width/height fitting and
                # the minimum readable font are the actual safety constraints.
                max_lines=None,
                balanced=balanced_layout,
            )
        if not fitted and expanded_bbox:
            fitted = fit_text(
                draw, translation, selected_font, width, height,
                fit_minimum, preferred_size, style_spacing, stroke_width,
                balanced=balanced_layout,
            )
        if not fitted:
            source_box = group["bbox"]
            # Tiny hesitation/SFX marks such as "Ooh..." are often OCR'd in a
            # narrow 80x60 box. Preserve them instead of creating a pointless
            # review when a readable replacement cannot fit.
            tiny_source = int(source_box["width"]) <= 100 and int(source_box["height"]) <= 70
            if (style == "sfx" and int(bbox["width"]) <= 80 and int(bbox["height"]) <= 50) or tiny_source:
                # Restore anything removal may have changed before preserving a
                # tiny SFX/noise mark; never leave an empty cleaned rectangle.
                for item in source_regions:
                    line = item["bbox"]
                    rx, ry = int(line["x"]), int(line["y"])
                    rw, rh = int(line["width"]), int(line["height"])
                    rendered.paste(Image.fromarray(reference[ry:ry + rh, rx:rx + rw]), (rx, ry))
                    allowed[ry:ry + rh, rx:rx + rw] = True
                results.append({
                    "id": group["id"], "status": "PRESERVED", "reason": "TINY_SFX_PRESERVED",
                    "bbox": source_box, "style": "sfx",
                })
                continue
            results.append({
                "id": group["id"], "status": "NEEDS_REVIEW",
                "reason": "TEXT_TOO_LARGE", "bbox": bbox,
            })
            continue
        # Some comic fonts report optimistic vertical metrics. Enforce a visual
        # density ceiling so 3+ lines never collide even when Pillow says they fit.
        if fitted["line_count"] >= 3 and fitted["font_size"] * fitted["line_count"] > height * 0.88:
            density_cap = max(fit_minimum, int(height / (fitted["line_count"] * 1.14)))
            denser_fit = fit_text(
                draw, translation, selected_font, width, height,
                fit_minimum, min(fitted["font_size"], density_cap),
                style_spacing, stroke_width,
                balanced=balanced_layout,
            )
            if denser_fit:
                fitted = denser_fit
        # Readability is a production constraint, including compact bubbles.
        # A small source surface is not permission to emit microscopic text.
        # Chat UI may be slightly denser, but never below half the RAW scale.
        minimum_scale = max(0.50, args.min_scale_ratio - 0.10) if style == "chat" else args.min_scale_ratio
        if compact_raw_text:
            minimum_scale = min(minimum_scale, 0.36)
        if layout_constrained:
            minimum_scale = min(minimum_scale, 0.32)
        if fitted["font_size"] < preferred_size * minimum_scale:
            # Cleanup has already removed RAW. Keep that clean surface and ask
            # for review; never silently restore English or export tiny text.
            bx, by = int(bbox["x"]), int(bbox["y"])
            allowed[by:by + int(bbox["height"]), bx:bx + int(bbox["width"])] = True
            results.append({
                "id": group["id"], "status": "NEEDS_REVIEW",
                "reason": "FONT_SCALE_TOO_SMALL", "bbox": bbox,
                "font_size": fitted["font_size"], "preferred_font_size": preferred_size,
            })
            continue

        text_bbox = fitted["text_bbox"]
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x + (width - text_width) / 2 - text_bbox[0]
        text_y = y + (height - text_height) / 2 - text_bbox[1]
        if expanded_bbox:
            # Keep the visual glyph block centred on the RAW OCR box. The
            # expanded bubble remains a safety boundary only; it must not pull
            # deliberately right/left-positioned dialogue elsewhere.
            # Keep every translation centred where its RAW text was—including
            # stacked rectangular cards. The detected physical bubble/card is
            # only the maximum width/height; clamp the RAW-centred block when
            # it would cross that outer black boundary.
            raw_center_x = int(raw_box["x"]) + int(raw_box["width"]) / 2
            raw_center_y = int(raw_box["y"]) + int(raw_box["height"]) / 2
            desired_x = raw_center_x - text_width / 2 - text_bbox[0]
            desired_y = raw_center_y - text_height / 2 - text_bbox[1]
            min_x, max_x = x - text_bbox[0], x + width - text_width - text_bbox[0]
            min_y, max_y = y - text_bbox[1], y + height - text_height - text_bbox[1]
            text_x = min(max(desired_x, min_x), max_x)
            text_y = min(max(desired_y, min_y), max_y)
        if promo_card:
            card_layer = Image.new("RGBA", rendered.size, (0, 0, 0, 0))
            card_draw = ImageDraw.Draw(card_layer)
            radius = max(6, round(min(width, height) * 0.10))
            card_draw.rounded_rectangle(
                (x, y, x + width, y + height), radius=radius,
                fill=(10, 14, 22, 232), outline=(255, 255, 255, 72),
                width=max(1, round(min(width, height) * 0.018)),
            )
            rendered.alpha_composite(card_layer) if rendered.mode == "RGBA" else rendered.paste(card_layer, (0, 0), card_layer)
            draw = ImageDraw.Draw(rendered)
            text_color, stroke_width, stroke_fill = (250, 250, 250), 0, (250, 250, 250)
            ax1, ay1 = max(0, x - 2), max(0, y - 2)
            ax2, ay2 = min(original.shape[1], x + width + 3), min(original.shape[0], y + height + 3)
            allowed[ay1:ay2, ax1:ax2] = True
        if args.hard_clip and rotation:
            exact = exact_raw_overlay(
                draw, translation, selected_font, preferred_size, fit_minimum,
                style_spacing, [], rotation, group["bbox"], None,
            )
            if not exact:
                results.append({"id": group["id"], "status": "NEEDS_REVIEW",
                                "reason": "RAW_POLYGON_TEXT_OVERFLOW", "bbox": group["bbox"]})
                continue
            exact_image, fitted, raw_surface, exact_x, exact_y = exact
            rendered.paste(exact_image, (exact_x, exact_y), exact_image)
            exact_allowed = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(exact_allowed, np.round(raw_surface).astype(np.int32), 255)
            allowed |= exact_allowed.astype(bool)
        elif tilted_surface:
            surface_width, surface_height = tilted_surface["width"], tilted_surface["height"]
            layer = Image.new("RGBA", (surface_width, surface_height), (0, 0, 0, 0))
            layer_draw = ImageDraw.Draw(layer)
            local_x = surface_margin_x + (width - text_width) / 2 - text_bbox[0]
            local_y = surface_margin_y + (height - text_height) / 2 - text_bbox[1]
            layer_draw.multiline_text(
                (local_x, local_y), fitted["wrapped"], font=fitted["font"],
                fill=text_color + (255,), spacing=fitted["spacing"], align="center",
                stroke_width=stroke_width, stroke_fill=stroke_fill + (255,),
            )
            rgba = np.asarray(layer)
            source_quad = np.float32([[0, 0], [surface_width - 1, 0],
                                      [surface_width - 1, surface_height - 1], [0, surface_height - 1]])
            destination_quad = tilted_surface["quad"].astype(np.float32)
            matrix = cv2.getPerspectiveTransform(source_quad, destination_quad)
            warped = cv2.warpPerspective(rgba, matrix, (original.shape[1], original.shape[0]),
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
            new_text_mask = warped[:, :, 3] > 20
            collision_pixels = int(np.count_nonzero(new_text_mask & occupied_text))
            if collision_pixels:
                results.append({
                    "id": group["id"], "status": "NEEDS_REVIEW",
                    "reason": "TYPESET_COLLISION", "bbox": group["bbox"],
                    "collision_pixels": collision_pixels,
                })
                continue
            warped_image = Image.fromarray(warped, mode="RGBA")
            rendered.paste(warped_image, (0, 0), warped_image)
            occupied_text |= new_text_mask
            polygon_mask = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(polygon_mask, np.round(destination_quad).astype(np.int32), 255)
            allowed |= polygon_mask.astype(bool)
        elif rotation:
            # Tilted RAW captions use the simplest lossless replacement: keep
            # the OCR rectangle's exact size, centre and angle, paint that same
            # rectangle white, then place the translation inside it. Nothing
            # outside the rotated source rectangle is cleaned or reconstructed.
            raw_box = raw_oriented_bounds(source_regions, rotation, group["bbox"])
            block_width = raw_box["width"]
            block_height = raw_box["height"]
            exact_fit = fit_text(
                draw, translation, selected_font,
                max(1, round(block_width * 0.88)), max(1, round(block_height * 0.82)),
                fit_minimum, preferred_size, style_spacing, 0,
            )
            if not exact_fit:
                results.append({
                    "id": group["id"], "status": "NEEDS_REVIEW",
                    "reason": "RAW_POLYGON_TEXT_OVERFLOW", "bbox": group["bbox"],
                })
                continue
            fitted = exact_fit
            text_bbox = fitted["text_bbox"]
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            # V2 preserves an existing bubble/card surface. Only the translated
            # glyphs are composited; cleanup already removed the RAW glyphs.
            # Legacy mode retains the opaque replacement for compatibility.
            layer_background = (
                (255, 255, 255, 255)
                if args.layout_engine == "legacy" and not args.hard_clip
                else (0, 0, 0, 0)
            )
            layer = Image.new("RGBA", (block_width, block_height), layer_background)
            layer_draw = ImageDraw.Draw(layer)
            local_text_x = (block_width - text_width) / 2 - text_bbox[0]
            local_text_y = (block_height - text_height) / 2 - text_bbox[1]
            layer_draw.multiline_text(
                (local_text_x, local_text_y), fitted["wrapped"], font=fitted["font"],
                fill=(12, 12, 12, 255), spacing=fitted["spacing"], align="center",
                stroke_width=0,
            )
            tilted = layer.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True)
            center_x, center_y = raw_box["center"]
            paste_x = round(center_x - tilted.width / 2)
            paste_y = round(center_y - tilted.height / 2)
            surface = surface_by_id.get(group["id"])
            if (args.hard_clip and surface and surface.get("safe_polygon")
                    and surface.get("surface_type") == "BUBBLE_FLAT"):
                tilted_rgba = np.asarray(tilted).copy()
                alpha_before_clip = int(np.count_nonzero(tilted_rgba[:, :, 3] > 20))
                surface_clip = np.zeros((tilted.height, tilted.width), dtype=np.uint8)
                local_surface = np.round(
                    np.asarray(surface["safe_polygon"], dtype=np.float32)
                    - np.array([paste_x, paste_y], dtype=np.float32)
                ).astype(np.int32)
                cv2.fillConvexPoly(surface_clip, local_surface, 255)
                tilted_rgba[:, :, 3] = cv2.bitwise_and(tilted_rgba[:, :, 3], surface_clip)
                alpha_after_clip = int(np.count_nonzero(tilted_rgba[:, :, 3] > 20))
                clip_loss_ratio = (
                    (alpha_before_clip - alpha_after_clip) / max(1, alpha_before_clip)
                )
                if clip_loss_ratio > 0.02:
                    if source_regions:
                        rx1 = max(0, min(int(item["bbox"]["x"]) for item in source_regions))
                        ry1 = max(0, min(int(item["bbox"]["y"]) for item in source_regions))
                        rx2 = min(reference.shape[1], max(
                            int(item["bbox"]["x"] + item["bbox"]["width"]) for item in source_regions
                        ))
                        ry2 = min(reference.shape[0], max(
                            int(item["bbox"]["y"] + item["bbox"]["height"]) for item in source_regions
                        ))
                        rendered.paste(Image.fromarray(reference[ry1:ry2, rx1:rx2]), (rx1, ry1))
                        allowed[ry1:ry2, rx1:rx2] = True
                    results.append({
                        "id": group["id"], "status": "NEEDS_REVIEW",
                        "reason": "SURFACE_CLIP_LOSS", "bbox": group["bbox"],
                        "clip_loss_ratio": round(clip_loss_ratio, 4),
                    })
                    continue
                tilted = Image.fromarray(tilted_rgba, mode="RGBA")
                surface_allowed_mask = np.zeros(original.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(
                    surface_allowed_mask,
                    np.round(np.asarray(surface["safe_polygon"])).astype(np.int32), 255,
                )
                allowed |= surface_allowed_mask.astype(bool)
            pending_tilted.append((center_y, tilted, paste_x, paste_y))
            allowed[max(0, paste_y):min(original.shape[0], paste_y + tilted.height),
                    max(0, paste_x):min(original.shape[1], paste_x + tilted.width)] = True
        else:
            surface = surface_by_id.get(group["id"])
            if (args.hard_clip and surface and surface.get("safe_polygon")
                    and surface.get("surface_type") == "BUBBLE_FLAT"):
                safe_polygon = np.asarray(surface["safe_polygon"], dtype=np.float32)
                px, py, pw, ph = cv2.boundingRect(np.round(safe_polygon).astype(np.int32))
                px, py = max(0, px), max(0, py)
                pw = min(original.shape[1] - px, pw)
                ph = min(original.shape[0] - py, ph)
                local_layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
                local_draw = ImageDraw.Draw(local_layer)
                local_draw.multiline_text(
                    (text_x - px, text_y - py), fitted["wrapped"], font=fitted["font"],
                    fill=text_color + (255,), spacing=fitted["spacing"], align="center",
                    stroke_width=stroke_width, stroke_fill=stroke_fill + (255,),
                )
                local_rgba = np.asarray(local_layer).copy()
                clip_mask = np.zeros((ph, pw), dtype=np.uint8)
                local_polygon = np.round(safe_polygon - np.array([px, py])).astype(np.int32)
                cv2.fillConvexPoly(clip_mask, local_polygon, 255)
                local_rgba[:, :, 3] = cv2.bitwise_and(local_rgba[:, :, 3], clip_mask)
                text_mask = local_rgba[:, :, 3] > 20
                collision = int(np.count_nonzero(text_mask & occupied_text[py:py + ph, px:px + pw]))
                if collision:
                    results.append({
                        "id": group["id"], "status": "NEEDS_REVIEW",
                        "reason": "TYPESET_COLLISION", "bbox": group["bbox"],
                        "collision_pixels": collision,
                    })
                    continue
                clipped_layer = Image.fromarray(local_rgba, mode="RGBA")
                rendered.paste(clipped_layer, (px, py), clipped_layer)
                occupied_text[py:py + ph, px:px + pw] |= text_mask
                safe_mask = np.zeros(original.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(safe_mask, np.round(safe_polygon).astype(np.int32), 255)
                allowed |= safe_mask.astype(bool)
            else:
                if stacked_text_card:
                    # Italic comic fonts have asymmetric left/top overhangs;
                    # deriving their origin from bbox edges makes the visible
                    # glyph block sit left and above the RAW centre. Draw from
                    # the already-clamped visual centre with a true middle
                    # anchor so the blue RAW box centre is respected.
                    visual_center_x = text_x + text_bbox[0] + text_width / 2
                    visual_center_y = text_y + text_bbox[1] + text_height / 2
                    draw.multiline_text(
                        (visual_center_x, visual_center_y), fitted["wrapped"],
                        font=fitted["font"], fill=text_color,
                        spacing=fitted["spacing"], align="center", anchor="mm",
                        stroke_width=stroke_width, stroke_fill=stroke_fill,
                    )
                else:
                    draw.multiline_text(
                        (text_x, text_y), fitted["wrapped"], font=fitted["font"],
                        fill=text_color, spacing=fitted["spacing"], align="center",
                        stroke_width=stroke_width, stroke_fill=stroke_fill,
                    )
                if stacked_text_card:
                    guard = 2
                    allowed[max(0, y - guard):min(original.shape[0], y + height + guard),
                            max(0, x - guard):min(original.shape[1], x + width + guard)] = True
                else:
                    allowed[y:y + height, x:x + width] = True
        results.append({
            "id": group["id"], "status": "TYPESET", "bbox": bbox,
            "font_size": fitted["font_size"], "line_count": fitted["line_count"],
            "source_line_height": round(source_line_height, 1),
            "preferred_font_size": preferred_size,
            "text_color": [int(value) for value in text_color],
            "stroke_width": stroke_width, "stroke_fill": [int(value) for value in stroke_fill],
            "artwork_background": artwork_background, "wrapped": fitted["wrapped"],
            "layout_engine": args.layout_engine,
            "flat_white_background": flat_white_background,
            "rectangular_caption": rectangular_caption,
            "style": style, "font": selected_font, "line_spacing_ratio": style_spacing,
            "rotation_degrees": rotation,
            "surface_type": "tilted_card" if tilted_surface else ("bubble" if expanded_bbox else "ocr_box"),
            "surface_polygon": tilted_surface["quad"].round(1).tolist() if tilted_surface else None,
            "surface_fill_ratio": tilted_surface["fill_ratio"] if tilted_surface else None,
            "expanded_to_bubble": bool(expanded_bbox),
            "expansion_source": expansion_source,
            "stacked_card_expansion_rejected": stacked_card_expansion_rejected,
            "stacked_text_card": stacked_text_card,
            "promo_card": promo_card,
            "quality_v2_surface": surface_by_id.get(group["id"]),
        })

    # Apply exact RAW cards last so cleanup/restoration from another group can
    # never erase them. Bottom cards are painted first, matching paper stacking.
    for _, tilted, paste_x, paste_y in sorted(pending_tilted, key=lambda item: item[0], reverse=True):
        rendered.paste(tilted, (paste_x, paste_y), tilted)
    output = np.asarray(rendered)
    changed = np.any(output != original, axis=2)
    outside = int(np.count_nonzero(changed & ~allowed))
    if outside:
        raise RuntimeError(f"Pixel safety failed: {outside} pixels changed outside typeset boxes")
    if output.shape != original.shape:
        raise RuntimeError("Output dimensions differ from cleaned input")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered.save(args.output, format="PNG", optimize=True)
    report = {
        "input_dimensions": [original.shape[1], original.shape[0]],
        "output_dimensions": [output.shape[1], output.shape[0]],
        "font": args.font,
        "layout_engine": args.layout_engine,
        "seconds": round(time.perf_counter() - started, 2),
        "changed_pixels": int(np.count_nonzero(changed)),
        "outside_typeset_box_changed_pixels": outside,
        "results": results,
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
