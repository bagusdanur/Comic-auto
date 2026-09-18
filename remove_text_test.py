#!/usr/bin/env python3
"""Phase 3: safely remove dark text only from flat, light comic bubbles."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hybrid_inpaint import create_lama
from xianscan_client import XianScanClient, allowed_mask as xianscan_allowed_mask, payload_for_regions


def source_promo_page(ocr: dict) -> bool:
    """Detect short scanlator ads so their fragments never enter dialogue inpainting."""
    regions=ocr.get('regions',[])
    joined=' '.join(str(r.get('text','')).upper() for r in regions)
    compact=re.sub(r'\s+','',joined)
    width=int(ocr.get('width') or 0);height=int(ocr.get('height') or 0)
    madara=(height and width and height <= width*.75
            and 'YOU CAN READ' in joined and 'CHAPTERS' in joined
            and bool(re.search(r'MADARA(?:SCAN[S5])?\.?C[O0][MN]',compact)))
    eva=(height and width and width*.9 <= height <= width*1.8
         and bool(re.search(r'E[VY]ASCAN[S5]',compact))
         and 'READ' in compact and ('ONLYAT' in compact or 'READONLY' in compact)
         and 'SUPPORT' in compact
         and ('REACHOUT' in compact or 'DISCORD' in compact))
    return bool(madara or eva)


def hybrid_artwork_eligible(source_credit: bool, surface: dict | None,
                            details: list[dict], mask: np.ndarray) -> bool:
    """Only sparse non-promotional artwork glyph masks may use generative cleanup."""
    ratios = [float(item.get("mask_ratio", 1.0)) for item in details]
    return bool(
        not source_credit
        and surface is not None
        and surface.get("surface_type") == "ARTWORK_TEXT"
        and ratios
        and max(ratios) <= 0.38
        and float(np.mean(mask > 0)) <= 0.08
    )


def reconstruct_flat_surface(image: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Remove a complete text box while preserving a bubble's local gradient."""
    x1, y1, x2, y2 = box
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    crop = image[y1:y2, x1:x2]
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0)
    border_gray = np.mean(border, axis=1)
    border_chroma = np.max(border, axis=1) - np.min(border, axis=1)
    # A large OpenCV inpaint over plain white balloons pulls black letter cores
    # inward and creates vertical shadows. Sample the already-existing clean
    # white edge and paint only the verified rectangular text area instead.
    if (float(np.mean(border_gray >= 220)) >= 0.84
            and float(np.median(border_chroma)) <= 10):
        clean = border[border_gray >= 220]
        fill = np.median(clean, axis=0).round().astype(np.uint8)
        restored = image.copy()

        # Never overwrite dark outline strokes (bubble borders) or colored artwork outside the bubble.
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        crop_chroma = (
            np.max(crop, axis=2) - np.min(crop, axis=2)
            if crop.ndim == 3 else np.zeros_like(crop_gray)
        )

        # Stroke/outline pixels have low gray (dark). Bubble interior is light (gray >= 180, low chroma).
        fill_mask = (crop_gray >= 180) & (crop_chroma <= 15)

        # Include dark letter glyphs (gray < 180) only if they don't connect to the crop boundary
        dark_pixels = (crop_gray < 180).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dark_pixels, connectivity=8)
        for label in range(1, num_labels):
            lx = stats[label, cv2.CC_STAT_LEFT]
            ly = stats[label, cv2.CC_STAT_TOP]
            lw = stats[label, cv2.CC_STAT_WIDTH]
            lh = stats[label, cv2.CC_STAT_HEIGHT]
            touches_edge = (
                lx <= 1 or ly <= 1
                or (lx + lw) >= crop.shape[1] - 1
                or (ly + lh) >= crop.shape[0] - 1
            )
            if not touches_edge:
                fill_mask[labels == label] = True

        restored_crop = crop.copy()
        restored_crop[fill_mask] = fill
        restored[y1:y2, x1:x2] = restored_crop

        painted_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        painted_mask[y1:y2, x1:x2] = fill_mask.astype(np.uint8) * 255
        return restored, painted_mask
    radius = max(3, min(9, round(min(x2 - x1, y2 - y1) * 0.08)))
    restored = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    return restored, mask


def padded_group_box(regions: list[dict], image_shape: tuple[int, ...],
                     surface: dict | None = None) -> tuple[int, int, int, int] | None:
    """Union fragmented OCR lines and include the outline/AA halo safely."""
    boxes = [item.get("bbox") for item in regions if item.get("bbox")]
    if not boxes:
        return None
    height, width = image_shape[:2]
    x1 = min(int(box["x"]) for box in boxes)
    y1 = min(int(box["y"]) for box in boxes)
    x2 = max(int(box["x"]) + int(box["width"]) for box in boxes)
    y2 = max(int(box["y"]) + int(box["height"]) for box in boxes)
    # Expand search bounds so wide letters, serifs, italics, and punctuation are fully included.
    pad_x = max(24, round((x2 - x1) * 0.15))
    pad_y = max(16, round((y2 - y1) * 0.25))
    x1, y1, x2, y2 = max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)
    boundary_poly = (surface or {}).get("polygon") or (surface or {}).get("safe_polygon") or []
    if boundary_poly:
        polygon = np.asarray(boundary_poly, dtype=np.float32)
        sx1, sy1 = np.floor(np.min(polygon, axis=0)).astype(int)
        sx2, sy2 = np.ceil(np.max(polygon, axis=0)).astype(int)
        x1, y1, x2, y2 = max(x1, sx1), max(y1, sy1), min(x2, sx2), min(y2, sy2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def locally_flat_text_surface(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    """Accept a bright low-chroma text patch even if the page-level map is uncertain."""
    x1, y1, x2, y2 = box
    crop = image[y1:y2, x1:x2]
    if not crop.size:
        return False
    gray = np.mean(crop, axis=2)
    chroma = np.max(crop, axis=2) - np.min(crop, axis=2)
    border = np.concatenate((crop[0], crop[-1], crop[:, 0], crop[:, -1]), axis=0)
    border_gray = np.mean(border, axis=1)
    return bool(np.mean(gray >= 205) >= 0.70 and np.median(chroma) <= 14 and np.mean(border_gray >= 205) >= 0.72)


def flat_bubble_residual_mask(image: np.ndarray, surface: dict,
                              limit_box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Select lettering missed by OCR while excluding the enclosing outline."""
    height, width = image.shape[:2]
    boundary_poly = (surface or {}).get("polygon") or (surface or {}).get("safe_polygon") or []
    polygon = np.asarray(boundary_poly, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) < 4:
        return mask
    cv2.fillConvexPoly(mask, np.round(polygon).astype(np.int32), 255)
    # Erode the bubble boundary conservatively to avoid the enclosing black stroke outline
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    if limit_box:
        local = np.zeros_like(mask)
        x1, y1, x2, y2 = limit_box
        local[y1:y2, x1:x2] = 255
        search_mask = cv2.bitwise_and(mask, local)
    else:
        search_mask = mask
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    candidate = np.where((gray < 190) & (search_mask > 0), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    glyphs = np.zeros_like(candidate)
    surface_area = max(1, int(np.count_nonzero(mask)))

    # Outer margin check to avoid touching the actual bubble stroke
    border_check = cv2.dilate(cv2.bitwise_not(mask), np.ones((3, 3), np.uint8))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_x = int(stats[label, cv2.CC_STAT_LEFT])
        component_y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        elongated = (component_width > max(60, component_height * 12) or 
                     component_height > max(60, component_width * 12))
        comp_mask = (labels[component_y:component_y+component_height, component_x:component_x+component_width] == label)
        nearby_border = border_check[component_y:component_y+component_height, component_x:component_x+component_width]
        touches_bubble_border = np.any(comp_mask & (nearby_border > 0))
        if not elongated and not touches_bubble_border and 2 <= area <= surface_area * 0.25:
            glyphs[labels == label] = 255
    glyphs = cv2.dilate(glyphs, np.ones((3, 3), np.uint8), iterations=2)
    return cv2.bitwise_and(glyphs, mask)


def inpaint_glyph_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct glyph pixels from their neighbourhood; never paint a card."""
    points = cv2.findNonZero(mask)
    if points is None:
        return image.copy()
    x, y, width, height = cv2.boundingRect(points)
    gutter = 8
    x1, y1 = max(0, x - gutter), max(0, y - gutter)
    x2, y2 = min(image.shape[1], x + width + gutter), min(image.shape[0], y + height + gutter)
    local_image = image[y1:y2, x1:x2]
    local_mask = mask[y1:y2, x1:x2]
    candidate = cv2.inpaint(local_image, local_mask, 3, cv2.INPAINT_TELEA)
    result = image.copy()
    selected = local_mask > 0
    result[y1:y2, x1:x2][selected] = candidate[selected]
    return result


def enclosing_bubble_surface(image: np.ndarray, box: tuple[int, int, int, int]) -> dict | None:
    """Recover a closed balloon interior when the surface map classified it as artwork."""
    x1, y1, x2, y2 = box
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    outline = np.where(gray < 105, 255, 0).astype(np.uint8)
    outline = cv2.morphologyEx(outline, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(outline, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_area = max(1, (x2 - x1) * (y2 - y1))
    candidates = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < raw_area * 1.3 or bw >= image.shape[1] * 0.98:
            continue
        if cv2.pointPolygonTest(contour, center, False) < 0:
            continue
        if not (0.9 <= bw / max(1, bh) <= 6.5):
            continue
        candidates.append((area, contour))
    if not candidates:
        return None
    contour = min(candidates, key=lambda item: item[0])[1]
    polygon = cv2.convexHull(contour).reshape(-1, 2).astype(float).tolist()
    return {"safe_polygon": polygon, "polygon": polygon}


def region_mask(image: np.ndarray, bbox: dict) -> tuple[np.ndarray | None, dict]:
    height, width = image.shape[:2]
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(width, x1 + int(bbox["width"]))
    y2 = min(height, y1 + int(bbox["height"]))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, {"reason": "EMPTY_BBOX"}

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    light_ratio = float(np.mean(gray >= 220))
    edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    edge_light_ratio = float(np.mean(edge >= 210))
    background = crop[gray >= 220]
    stats = {
        "light_ratio": round(light_ratio, 3),
        "edge_light_ratio": round(edge_light_ratio, 3),
    }
    if light_ratio < 0.55 or edge_light_ratio < 0.62 or len(background) < 20:
        return None, {**stats, "reason": "COMPLEX_BACKGROUND"}

    fill_rgb = np.median(background, axis=0).round().astype(np.uint8)
    # OCR boxes tightly surround the glyphs. Select dark glyph pixels inside that
    # box, then expand by one pixel for anti-aliased edges without touching the
    # rest of the speech bubble.
    # Detect the solid glyph core conservatively. JPEG halos are handled by
    # spatial expansion below; raising this threshold too far can connect words
    # through compression noise and make the component filter reject them.
    dark = (gray < 210).astype(np.uint8) * 255
    count, labels, component_stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    glyphs = np.zeros_like(dark)
    crop_area = gray.shape[0] * gray.shape[1]
    for label in range(1, count):
        area = int(component_stats[label, cv2.CC_STAT_AREA])
        component_x = int(component_stats[label, cv2.CC_STAT_LEFT])
        component_y = int(component_stats[label, cv2.CC_STAT_TOP])
        component_width = int(component_stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(component_stats[label, cv2.CC_STAT_HEIGHT])
        touches_edge = (component_x <= 1 or component_y <= 1
                        or component_x + component_width >= crop.shape[1] - 1
                        or component_y + component_height >= crop.shape[0] - 1)
        elongated_line = (component_width > max(14, component_height * 9)
                          or component_height > max(18, component_width * 12))
        if not elongated_line and 2 <= area <= crop_area * 0.22 \
                and component_width < crop.shape[1] * 0.8 and component_height < crop.shape[0] * 0.9:
            glyphs[labels == label] = 255
    glyphs = cv2.dilate(glyphs, np.ones((3, 3), np.uint8), iterations=2)
    # A hard inset is the final guard against painting across a bubble/panel
    # outline that happens to pass along the edge of a tight OCR rectangle.
    if glyphs.shape[0] > 4 and glyphs.shape[1] > 4:
        glyphs[:2, :] = 0
        glyphs[-2:, :] = 0
        glyphs[:, :2] = 0
        glyphs[:, -2:] = 0
    if not np.any(glyphs):
        return None, {**stats, "reason": "NO_SAFE_GLYPHS"}

    full = np.zeros((height, width), dtype=np.uint8)
    full[y1:y2, x1:x2] = glyphs
    return full, {
        **stats,
        "reason": "SAFE_FLAT_BACKGROUND",
        "fill_rgb": fill_rgb.tolist(),
        "masked_pixels": int(np.count_nonzero(glyphs)),
    }


def artwork_text_mask(image: np.ndarray, bbox: dict) -> tuple[np.ndarray | None, dict]:
    """Conservative glyph-shaped mask for text printed over artwork."""
    height, width = image.shape[:2]
    x1 = max(0, int(bbox["x"]))
    y1 = max(0, int(bbox["y"]))
    x2 = min(width, x1 + int(bbox["width"]))
    y2 = min(height, y1 + int(bbox["height"]))
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None, {"reason": "EMPTY_BBOX"}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    # OCR boxes are tight around a text line. Local contrast plus edges selects
    # glyph strokes while avoiding a rectangular edit over the illustration.
    background = cv2.medianBlur(gray, 15 if min(gray.shape) >= 15 else max(3, min(gray.shape) // 2 * 2 - 1))
    contrast = cv2.absdiff(gray, background)
    edges = cv2.Canny(gray, 45, 140)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    # Most overlay captions use a dark glyph core surrounded by a white or
    # colored outline. Anchor on that dark core, then expand into its outline;
    # selecting all bright pixels would also select walls, shirts, and faces.
    candidate = (((gray <= 115) & (contrast >= 10)) | ((gray <= 155) & (contrast >= 30) & (edges > 0))).astype(np.uint8) * 255
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    glyphs = np.zeros_like(candidate)
    area_limit = gray.size * 0.35
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if 2 <= area <= area_limit and component_width < crop.shape[1] * 0.92 and component_height < crop.shape[0] * 0.95:
            glyphs[labels == label] = 255
    # Grow from the dark letter cores into a bright/contrasting outline only.
    # An unconditional dilation easily consumes most of a tight OCR rectangle,
    # especially for multi-line captions over detailed artwork.
    kernel = np.ones((3, 3), np.uint8)
    outline_allowed = ((gray >= 175) | (contrast >= 24)).astype(np.uint8) * 255
    for _ in range(3):
        border = cv2.subtract(cv2.dilate(glyphs, kernel, iterations=1), glyphs)
        glyphs = cv2.bitwise_or(glyphs, cv2.bitwise_and(border, outline_allowed))
    # Captions frequently have a colored drop-shadow several pixels beyond the
    # bright glyph. Saturated source boxes need a wider halo than monochrome
    # letters, otherwise the pink/red shadow bleeds through the replacement.
    chroma = np.max(crop, axis=2).astype(np.int16) - np.min(crop, axis=2).astype(np.int16)
    halo_iterations = 9 if float(np.mean(chroma > 18)) >= 0.12 else 4
    glyphs = cv2.dilate(glyphs, kernel, iterations=halo_iterations)
    ratio = float(np.mean(glyphs > 0))
    if ratio >= 0.38:
        # A coloured/high-detail panel can make the wide halo absorb speed
        # lines, hair, or a wall edge until most of the OCR rectangle becomes
        # a mask.  Reconstructing that rectangle creates an obvious blurred
        # patch. Fall back to the detected glyph cores with a small AA halo;
        # this keeps cleanup local to the lettering.
        # Five pixels is enough to include the common white outline/drop
        # shadow; two left bright clouds around the removed dark core.
        glyphs = cv2.dilate(candidate, kernel, iterations=5)
        ratio = float(np.mean(glyphs > 0))
    if ratio < 0.008:
        return None, {"reason": "ARTWORK_MASK_UNCERTAIN", "mask_ratio": round(ratio, 3)}
    full = np.zeros((height, width), dtype=np.uint8)
    full[y1:y2, x1:x2] = glyphs
    return full, {"reason": "ARTWORK_GLYPH_INPAINT", "mask_ratio": round(ratio, 3), "masked_pixels": int(np.count_nonzero(glyphs))}


def reconstruct_dense_caption(image: np.ndarray, bbox: dict) -> tuple[np.ndarray, np.ndarray]:
    """Remove a dense artwork caption without inventing an opaque card.

    Each image column is interpolated between untouched bands immediately above
    and below the caption. This keeps vertical gradients, speed lines, and dark
    panel texture continuous while removing the complete outlined RAW glyph.
    """
    height, width = image.shape[:2]
    pad = max(6, min(14, round(min(int(bbox["width"]), int(bbox["height"])) * 0.06)))
    raw_x1 = max(0, int(bbox["x"]))
    raw_y1 = max(0, int(bbox["y"]))
    raw_x2 = min(width, int(bbox["x"] + bbox["width"]))
    raw_y2 = min(height, int(bbox["y"] + bbox["height"]))
    raw_crop = image[raw_y1:raw_y2, raw_x1:raw_x2]
    chroma = np.max(raw_crop, axis=2) - np.min(raw_crop, axis=2) if raw_crop.size else np.empty(0)
    saturated_glyph = bool(chroma.size and float(np.mean(chroma >= 70)) >= 0.18)
    pad_y = max(pad, round(int(bbox["height"]) * 0.45)) if saturated_glyph else pad
    x1 = max(0, int(bbox["x"]) - pad)
    y1 = max(0, int(bbox["y"]) - pad_y)
    x2 = min(width, int(bbox["x"] + bbox["width"]) + pad)
    y2 = min(height, int(bbox["y"] + bbox["height"]) + pad_y)
    band = max(2, min(6, pad // 2))
    top = np.median(image[max(0, y1 - band):y1 or 1, x1:x2], axis=0)
    bottom = np.median(image[y2:min(height, y2 + band), x1:x2], axis=0)
    if top.ndim != 2:
        top = image[max(0, y1 - 1), x1:x2].astype(np.float32)
    if bottom.ndim != 2:
        bottom = image[min(height - 1, y2), x1:x2].astype(np.float32)
    left = np.median(image[y1:y2, max(0, x1 - band):x1 or 1], axis=1)
    right = np.median(image[y1:y2, x2:min(width, x2 + band)], axis=1)
    if left.ndim != 2:
        left = image[y1:y2, max(0, x1 - 1)].astype(np.float32)
    if right.ndim != 2:
        right = image[y1:y2, min(width - 1, x2)].astype(np.float32)
    # Use the opposite border pair that agrees most closely. Brightness alone
    # chose a dark character below a flat label and produced a vertical smear.
    vertical_difference = float(np.mean(np.abs(top.astype(np.float32) - bottom.astype(np.float32))))
    horizontal_difference = float(np.mean(np.abs(left.astype(np.float32) - right.astype(np.float32))))
    if horizontal_difference <= vertical_difference:
        span = max(1, x2 - x1)
        weights = np.linspace(0.0, 1.0, span, dtype=np.float32)[None, :, None]
        fill = left[:, None, :] * (1.0 - weights) + right[:, None, :] * weights
    else:
        span = max(1, y2 - y1)
        weights = np.linspace(0.0, 1.0, span, dtype=np.float32)[:, None, None]
        fill = top[None, :, :] * (1.0 - weights) + bottom[None, :, :] * weights
    result = image.copy()
    result[y1:y2, x1:x2] = np.clip(fill, 0, 255).astype(np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return result, mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("ocr_json", type=Path)
    parser.add_argument("translations_json", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/cleaned.png"))
    parser.add_argument("--mask", type=Path, default=Path("output/mask-debug.png"))
    parser.add_argument("--report", type=Path, default=Path("output/removal-report.json"))
    parser.add_argument("--surface-map", type=Path)
    parser.add_argument("--hybrid-mode", choices=("off", "shadow", "on", "xianscan", "xianscan-full", "xianscan-mask", "xianscan-v3"), default="off")
    parser.add_argument("--lama-model", type=Path)
    parser.add_argument("--hybrid-report", type=Path)
    parser.add_argument("--xianscan-url", default="http://127.0.0.1:8123")
    parser.add_argument("--strict-artwork-inpaint", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()

    with Image.open(args.image) as source:
        original = np.asarray(source.convert("RGB")).copy()
    output = original.copy()
    model_path = args.lama_model
    hybrid_runtime_available = bool(
        model_path and model_path.is_file() and importlib.util.find_spec("onnxruntime") is not None
    )
    # Shadow mode must not double page latency or allocate the ~1 GB LaMa
    # session. It records eligibility; explicit offline pilots perform inference.
    lama = None
    hybrid_events = []
    xianscan = XianScanClient(args.xianscan_url) if args.hybrid_mode in (
        "xianscan", "xianscan-full", "xianscan-mask", "xianscan-v3"
    ) else None
    allowed_mask = np.zeros(original.shape[:2], dtype=np.uint8)
    ocr = json.loads(args.ocr_json.read_text(encoding="utf-8"))
    translations = json.loads(args.translations_json.read_text(encoding="utf-8"))
    whole_page_source_promo = source_promo_page(ocr)
    from branding_v52 import stacked_source_promo_header
    source_header_bottom = stacked_source_promo_header(original, ocr.get("regions", []))
    surface_by_id = {}
    if args.surface_map and args.surface_map.exists():
        surface_data = json.loads(args.surface_map.read_text(encoding="utf-8"))
        surface_by_id = {item["id"]: item for item in surface_data.get("surfaces", [])}
    by_id = {region["id"]: region for region in ocr["regions"]}
    results = []

    for group in translations["groups"]:
        source_normalized = " ".join(str(group.get("text", "")).upper().split())
        translation_normalized = " ".join(str(group.get("translation", "")).upper().split())
        group_top=int(group.get("bbox",{}).get("y",original.shape[0]))
        source_credit = whole_page_source_promo or (source_header_bottom and group_top < source_header_bottom) or bool(re.search(
            r"(?:SCAN(?:S|LATION)?|THUNDER|MADARA|EVASCANS?|READ\s+(?:AT|ONLY)|YOU\s+CAN\s+READ|HTTPS?://|WWW\.|\.(?:COM|ORG|NET)\b)",
            source_normalized,
        ))
        if source_normalized and source_normalized == translation_normalized and not source_credit:
            results.append({
                "id": group["id"], "status": "PRESERVED",
                "reason": "SOURCE_EQUALS_TRANSLATION",
            })
            continue
        surface = surface_by_id.get(group["id"])
        if args.strict_artwork_inpaint and source_credit:
            # Raw branding is handled exclusively by the conservative branding
            # stage, never by a dialogue-cleaning fallback with broader masks.
            results.append({'id':group['id'],'status':'PRESERVED','reason':'DEFER_SOURCE_BRANDING'})
            continue
        if args.strict_artwork_inpaint and not source_credit:
            if not surface:
                raise RuntimeError(f"V5 missing surface geometry: {group['id']}")
            if xianscan is None:
                raise RuntimeError("V5 artwork cleanup requires the inpainting service")
            source_regions = [by_id[r] for r in group["source_region_ids"] if r in by_id]
            if not source_regions:
                raise RuntimeError(f"V5 missing source regions: {group['id']}")
            # Structured RPG/system windows have a flat or translucent card
            # behind several labelled lines. Large generative rectangle masks
            # can leave long black streaks on these fields. Remove the actual
            # neutral glyph ink with deterministic local inpainting, while
            # coloured window borders remain outside every OCR-line mask.
            system_markers = sum(bool(re.search(pattern, source_normalized)) for pattern in (
                r'\bRANK\s*:', r'\bDESCRIPTION\s*:', r'\bMATERIAL\b',
                r'\bEQUIPMENT\b', r'\bSKILL\b',
            ))
            if system_markers >= 2:
                permitted = np.zeros(original.shape[:2], dtype=np.uint8)
                for region in source_regions:
                    box = region['bbox']
                    x, y, w, h = (int(box[key]) for key in ('x', 'y', 'width', 'height'))
                    pad_x = max(6, round(h * .24)); pad_y = max(5, round(h * .18))
                    x1, y1 = max(0, x-pad_x), max(0, y-pad_y)
                    x2, y2 = min(original.shape[1], x+w+pad_x), min(original.shape[0], y+h+pad_y)
                    crop = original[y1:y2, x1:x2]
                    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                    chroma = crop.max(axis=2).astype(np.int16) - crop.min(axis=2).astype(np.int16)
                    ink = ((gray < 155) & (chroma < 58)).astype(np.uint8) * 255
                    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
                    ink = cv2.dilate(ink, np.ones((7, 7), np.uint8), iterations=1)
                    permitted[y1:y2, x1:x2] = cv2.bitwise_or(permitted[y1:y2, x1:x2], ink)
                if np.any(permitted):
                    before = output.copy()
                    bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                    output = cv2.cvtColor(cv2.inpaint(bgr, permitted, 5, cv2.INPAINT_TELEA), cv2.COLOR_BGR2RGB)
                    delta = np.any(output != before, axis=2)
                    if np.any(delta & (permitted == 0)):
                        raise RuntimeError('V5 system-window cleanup changed pixels outside its mask')
                    allowed_mask |= np.where(delta, 255, 0).astype(np.uint8)
                    results.append({
                        "id": group["id"], "status": "CLEANED",
                        "surface_type": surface.get('surface_type'),
                        "removal_backend": "v5-system-window-neutral-telea",
                        "changed_pixels": int(np.count_nonzero(delta)),
                        "outside_mask_changed_pixels": 0,
                    })
                    continue
            # Include outline/shadow pixels that dark-glyph detectors miss.
            # Inpaint the source footprints; never paint an opaque text card.
            region_masks=[]
            for region in source_regions:
                box = region['bbox']
                x,y,w,h = (int(box[k]) for k in ('x','y','width','height'))
                pad = max(4, min(9, round(h * .16)))
                one=np.zeros(original.shape[:2],dtype=np.uint8)
                one[max(0,y-pad):min(original.shape[0],y+h+pad), max(0,x-pad):min(original.shape[1],x+w+pad)] = 255
                region_masks.append(one)
            # The sidecar intentionally rejects masks above 20% of a page.
            # Dense lore/credit pages can contain many safe OCR line boxes;
            # pack those boxes into bounded requests instead of weakening the
            # sidecar limit or aborting the entire chapter.
            batches=[];current=np.zeros(original.shape[:2],dtype=np.uint8)
            for one in region_masks:
                merged=cv2.bitwise_or(current,one)
                if np.any(current) and float(np.mean(merged>0))>.18:
                    batches.append(current);current=one
                else:current=merged
            if np.any(current):batches.append(current)
            total_changed=0
            for permitted in batches:
                if float(np.mean(permitted>0))>.18:
                    raise RuntimeError(f"V5 individual source footprint is implausibly large: {group['id']}")
                candidate = xianscan.inpaint_mask(output, permitted)
                delta = np.any(candidate != output, axis=2)
                if np.any(delta & (permitted == 0)):
                    raise RuntimeError("V5 artwork cleanup changed pixels outside its mask")
                output = candidate
                total_changed+=int(np.count_nonzero(delta))
                allowed_mask |= np.where(delta, 255, 0).astype(np.uint8)
            results.append({"id": group["id"], "status": "INPAINTED", "surface_type": surface.get('surface_type'), "removal_backend": "v5-source-footprint-lama-batched", "inpaint_batches":len(batches), "changed_pixels":total_changed, "outside_mask_changed_pixels": 0})
            continue
        surface_clip = None
        if surface and surface.get("polygon"):
            surface_clip = np.zeros(original.shape[:2], dtype=np.uint8)
            cv2.fillConvexPoly(surface_clip, np.round(surface["polygon"]).astype(np.int32), 255)
        group_mask = np.zeros(original.shape[:2], dtype=np.uint8)
        details = []
        safe = True
        source_regions = [by_id[item] for item in group.get("source_region_ids", []) if item in by_id]
        if args.hybrid_mode == "xianscan-v3" and not source_credit:
            cleanup_box = padded_group_box(source_regions, original.shape, surface)
            if cleanup_box and (surface or {}).get("surface_type") == "BUBBLE_FLAT":
                before = output.copy()
                exact_mask = flat_bubble_residual_mask(output, surface, cleanup_box)
                output = inpaint_glyph_mask(output, exact_mask)
                delta = np.any(output != before, axis=2)
                allowed_mask = cv2.bitwise_or(allowed_mask, exact_mask)
                results.append({
                    "id": group["id"], "status": "CLEANED",
                    "surface_type": (surface or {}).get("surface_type"),
                    "removal_backend": "v3-flat-residual-mask",
                    "cleanup_box": [int(value) for value in cleanup_box],
                    "changed_pixels": int(np.count_nonzero(delta)),
                })
                continue
            if cleanup_box and locally_flat_text_surface(output, cleanup_box):
                detected_bubble = enclosing_bubble_surface(output, cleanup_box)
                exact_mask = flat_bubble_residual_mask(output, detected_bubble, cleanup_box) if detected_bubble else None
                if exact_mask is not None and np.any(exact_mask):
                    before = output.copy()
                    output = inpaint_glyph_mask(output, exact_mask)
                    delta = np.any(output != before, axis=2)
                    allowed_mask = cv2.bitwise_or(allowed_mask, exact_mask)
                    results.append({
                        "id": group["id"], "status": "CLEANED",
                        "surface_type": (surface or {}).get("surface_type"),
                        "removal_backend": "v3-padded-glyph-mask",
                        "cleanup_box": [int(value) for value in cleanup_box],
                        "changed_pixels": int(np.count_nonzero(delta)),
                        "detected_bubble": True,
                    })
                    continue
        # Diagnostic/full mode deliberately routes every translatable surface
        # through XianScan, including flat speech bubbles. Production hybrid
        # mode keeps the faster deterministic flat-fill path below.
        if args.hybrid_mode == "xianscan-full" and xianscan is not None and not source_credit:
            try:
                before_xianscan = output.copy()
                candidate = xianscan.clean(output, payload_for_regions(source_regions))
                permitted = xianscan_allowed_mask(original.shape[:2], source_regions)
                delta = np.any(candidate != before_xianscan, axis=2)
                outside = int(np.count_nonzero(delta & (permitted == 0)))
                if outside:
                    raise RuntimeError(f"XianScan changed {outside} pixels outside source polygons")
                # Full mode intentionally accepts XianScan's complete patch.
                # The sidecar is still constrained to the source OCR polygons
                # and the outside-mask guard above rejects any leaked pixels.
                output = candidate
                changed_mask = np.where(delta, 255, 0).astype(np.uint8)
                allowed_mask = cv2.bitwise_or(allowed_mask, changed_mask)
                event = {
                    "id": group["id"], "surface_type": surface.get("surface_type") if surface else None,
                    "mode": args.hybrid_mode, "available": True,
                    "backend": "xianscan-lama-onnx", "changed_pixels": int(np.count_nonzero(delta)),
                    "changed_outside_mask": outside,
                }
                hybrid_events.append(event)
                results.append({
                    "id": group["id"], "status": "INPAINTED",
                    "surface_type": surface.get("surface_type") if surface else None,
                    "removal_backend": "xianscan-lama-onnx-full",
                })
                continue
            except Exception as error:
                hybrid_events.append({
                    "id": group["id"], "mode": args.hybrid_mode, "available": False,
                    "backend": "xianscan-lama-onnx", "error": str(error),
                    "fallback": "standard-surface-cleanup",
                })
        # Existing bright tilted cards need no inpainting and no replacement
        # background. Repaint only each RAW line polygon with the card's own
        # sampled interior colour, preserving the original card geometry.
        surface_stats = (surface or {}).get("stats") or {}
        tilted_is_verified_flat_card = bool(
            surface
            and abs(float(surface.get("angle_degrees", 0))) >= 2.5
            and float(surface_stats.get("light_ratio", 0)) >= 0.72
            and float(surface_stats.get("dark_ratio", 1)) <= 0.16
            and float(np.median(surface_stats.get("median_rgb") or [0])) >= 225
        )
        if tilted_is_verified_flat_card:
            line_operations = []
            tilted_safe = True
            for region_id in group["source_region_ids"]:
                region = by_id[region_id]
                polygon = np.asarray(region.get("polygon") or [], dtype=np.float32)
                if len(polygon) < 4:
                    tilted_safe = False
                    break
                center = np.mean(polygon, axis=0)
                expanded = center + (polygon - center) * 1.04
                line_mask = np.zeros(original.shape[:2], dtype=np.uint8)
                cv2.fillConvexPoly(line_mask, np.round(expanded).astype(np.int32), 255)
                if surface_clip is not None:
                    line_mask = cv2.bitwise_and(line_mask, surface_clip)
                pixels = original[line_mask > 0]
                if not len(pixels):
                    tilted_safe = False
                    break
                gray_pixels = np.mean(pixels, axis=1)
                light_pixels = pixels[gray_pixels >= 180]
                light_ratio = len(light_pixels) / len(pixels)
                if light_ratio < 0.42 or len(light_pixels) < 12:
                    stats = surface.get("stats") or {}
                    colored_flat_label = bool(
                        surface.get("surface_type") == "ARTWORK_TEXT"
                        and float(surface.get("confidence", 0)) >= 0.60
                        and float(stats.get("color_ratio", 0)) >= 0.45
                        and float(stats.get("dark_ratio", 1)) <= 0.35
                        and float(np.median(gray_pixels)) >= 110
                    )
                    if not colored_flat_label:
                        tilted_safe = False
                        break
                    # Colored paper/card: discard the darkest letter cores and
                    # sample the actual local paper colour from the upper half.
                    cutoff = float(np.percentile(gray_pixels, 48))
                    light_pixels = pixels[gray_pixels >= cutoff]
                    light_ratio = len(light_pixels) / len(pixels)
                fill = np.median(light_pixels, axis=0).round().astype(np.uint8)
                line_operations.append((line_mask, fill, region_id, light_ratio))
            if tilted_safe and line_operations:
                painted = np.zeros(original.shape[:2], dtype=np.uint8)
                for line_mask, fill, _, _ in line_operations:
                    output[line_mask > 0] = fill
                    painted = cv2.bitwise_or(painted, line_mask)
                allowed_mask = cv2.bitwise_or(allowed_mask, painted)
                results.append({
                    "id": group["id"], "status": "CLEANED",
                    "surface_type": surface.get("surface_type"),
                    "reason": "TILTED_EXISTING_SURFACE_FILL",
                    "regions": [{"region_id": region_id, "light_ratio": round(ratio, 3)}
                                for _, _, region_id, ratio in line_operations],
                })
                continue
        for region_id in group["source_region_ids"]:
            region = by_id[region_id]
            mask, detail = region_mask(original, region["bbox"])
            details.append({"region_id": region_id, **detail})
            if mask is None:
                safe = False
            else:
                group_mask = cv2.bitwise_or(group_mask, mask)
        if surface_clip is not None:
            group_mask = cv2.bitwise_and(group_mask, surface_clip)
        if not safe:
            # Some colored/gradient speech bubbles are bright overall, but one
            # tight OCR line fails the per-line edge test. If the expanded
            # complete group is still surrounded by a bright, uniform bubble,
            # clearing the grouped text rectangle is safer than leaving it for
            # review or placing Indonesian over the English glyphs.
            group_box = group["bbox"]
            group_pad = max(12, min(32, round(min(int(group_box["width"]), int(group_box["height"])) * 0.15)))
            gx1 = max(0, int(group_box["x"]) - group_pad)
            gy1 = max(0, int(group_box["y"]) - group_pad)
            gx2 = min(original.shape[1], int(group_box["x"] + group_box["width"]) + group_pad)
            gy2 = min(original.shape[0], int(group_box["y"] + group_box["height"]) + group_pad)
            group_crop = original[gy1:gy2, gx1:gx2]
            group_gray = cv2.cvtColor(group_crop, cv2.COLOR_RGB2GRAY)
            group_edge = np.concatenate((group_gray[0], group_gray[-1], group_gray[:, 0], group_gray[:, -1]))
            if np.mean(group_gray >= 190) >= 0.62 and np.mean(group_edge >= 190) >= 0.84:
                before_fill = output.copy()
                candidate, painted_mask = reconstruct_flat_surface(output, (gx1, gy1, gx2, gy2))
                # The reconstruction box surrounds RAW glyphs, while the V2
                # polygon describes the actual oval/balloon. Never let a
                # rectangular white fill escape through curved shoulders.
                if surface_clip is not None:
                    painted_mask = cv2.bitwise_and(painted_mask, surface_clip)
                    output[painted_mask > 0] = candidate[painted_mask > 0]
                    output[painted_mask == 0] = before_fill[painted_mask == 0]
                else:
                    output = candidate
                allowed_mask = cv2.bitwise_or(allowed_mask, painted_mask)
                results.append({"id": group["id"], "status": "CLEANED", "regions": details, "group_clear": True})
                continue
            artwork_mask = np.zeros(original.shape[:2], dtype=np.uint8)
            artwork_details = []
            for region_id in group["source_region_ids"]:
                region = by_id[region_id]
                mask, detail = artwork_text_mask(original, region["bbox"])
                artwork_details.append({"region_id": region_id, **detail})
                if mask is None:
                    artwork_mask = np.zeros_like(artwork_mask)
                    break
                artwork_mask = cv2.bitwise_or(artwork_mask, mask)
            if surface_clip is not None:
                artwork_mask = cv2.bitwise_and(artwork_mask, surface_clip)
            if not np.any(artwork_mask):
                results.append({"id": group["id"], "status": "NEEDS_REVIEW", "regions": artwork_details})
                continue
            dense_ratio = max(float(item.get("mask_ratio", 0)) for item in artwork_details)
            # Never synthesize the complete OCR rectangle on comic artwork.
            # Even when lettering is dense, doing so turns real panel texture
            # into a visibly blurred block. artwork_text_mask has already
            # reduced an overgrown mask back to conservative glyph cores.
            # Navier-Stokes continues nearby colour/edge directions through
            # lettering masks more smoothly. TELEA produced pale mosaic blobs
            # around outlined diagonal captions on coloured artwork.
            hybrid_event = {
                "id": group["id"], "surface_type": surface.get("surface_type") if surface else None,
                "mode": args.hybrid_mode, "mask_pixels": int(np.count_nonzero(artwork_mask)),
            }
            source_regions = [by_id[region_id] for region_id in group["source_region_ids"] if region_id in by_id]
            if xianscan is not None and source_regions and not source_credit:
                try:
                    before_xianscan = output.copy()
                    exact_mask_safe = bool(
                        args.hybrid_mode == "xianscan-mask"
                        and dense_ratio <= 0.38
                        and float(np.mean(artwork_mask > 0)) <= 0.08
                    )
                    if exact_mask_safe:
                        candidate = xianscan.inpaint_mask(output, artwork_mask)
                        permitted = artwork_mask
                        backend = "xianscan-lama-onnx-exact-mask"
                    else:
                        candidate = xianscan.clean(output, payload_for_regions(source_regions))
                        permitted = xianscan_allowed_mask(original.shape[:2], source_regions)
                        backend = (
                            "xianscan-lama-onnx-polygon-fallback"
                            if args.hybrid_mode == "xianscan-mask"
                            else "xianscan-lama-onnx"
                        )
                    delta = np.any(candidate != before_xianscan, axis=2)
                    outside = int(np.count_nonzero(delta & (permitted == 0)))
                    if outside:
                        raise RuntimeError(f"XianScan changed {outside} pixels outside source polygons")
                    output = candidate
                    changed_mask = np.where(delta, 255, 0).astype(np.uint8)
                    allowed_mask = cv2.bitwise_or(allowed_mask, changed_mask)
                    hybrid_event.update({
                        "available": True, "backend": backend,
                        "changed_pixels": int(np.count_nonzero(delta)),
                        "changed_outside_mask": outside,
                        "exact_mask_eligible": exact_mask_safe,
                    })
                    hybrid_events.append(hybrid_event)
                    results.append({
                        "id": group["id"], "status": "INPAINTED", "regions": artwork_details,
                        "surface_type": surface.get("surface_type") if surface else None,
                        "removal_backend": backend,
                    })
                    continue
                except Exception as error:
                    hybrid_event.update({
                        "available": False, "backend": "xianscan-lama-onnx",
                        "error": str(error), "fallback": "opencv-ns",
                    })
            hybrid_safe = hybrid_artwork_eligible(source_credit, surface, artwork_details, artwork_mask)
            hybrid_event.update({
                "runtime_available": hybrid_runtime_available,
                "eligible": hybrid_safe,
                "eligibility_reason": "SAFE_ARTWORK_GLYPH_MASK" if hybrid_safe else "UNSAFE_OR_DENSE_MASK",
            })
            if args.hybrid_mode == "on" and hybrid_safe:
                if lama is None:
                    lama = create_lama(model_path, threads=2)
            if lama is not None and hybrid_safe:
                try:
                    hybrid_result = lama(output, artwork_mask)
                    hybrid_event.update({
                        "available": True, "backend": hybrid_result.backend,
                        "patch_count": hybrid_result.patch_count,
                        "changed_outside_mask": hybrid_result.changed_outside_mask,
                    })
                    if args.hybrid_mode == "on":
                        output = hybrid_result.image
                        removal_backend = hybrid_result.backend
                    else:
                        output = cv2.inpaint(output, artwork_mask, 3, cv2.INPAINT_NS)
                        removal_backend = "opencv-ns-shadow-primary"
                except Exception as error:
                    hybrid_event.update({"available": True, "error": str(error), "fallback": "opencv-ns"})
                    output = cv2.inpaint(output, artwork_mask, 3, cv2.INPAINT_NS)
                    removal_backend = "opencv-ns-fallback"
            else:
                hybrid_event.update({
                    "available": hybrid_runtime_available,
                    "fallback": "opencv-ns",
                    "reason": "SHADOW_NO_INFERENCE" if args.hybrid_mode == "shadow" else "HYBRID_SAFETY_GATE",
                })
                output = cv2.inpaint(output, artwork_mask, 3, cv2.INPAINT_NS)
                removal_backend = "opencv-ns-fallback"
            hybrid_events.append(hybrid_event)
            allowed_mask = cv2.bitwise_or(allowed_mask, artwork_mask)
            results.append({"id": group["id"], "status": "INPAINTED", "regions": artwork_details,
                            "surface_type": surface.get("surface_type") if surface else None,
                            "removal_backend": removal_backend})
            continue

        # When the complete grouped area is safely surrounded by a light flat
        # surface, clear it as one unit. This catches letters touching OCR-box
        # edges that conservative connected-component filtering must reject.
        group_box = group["bbox"]
        group_pad = max(12, min(32, round(min(int(group_box["width"]), int(group_box["height"])) * 0.15)))
        gx1 = max(0, int(group_box["x"]) - group_pad)
        gy1 = max(0, int(group_box["y"]) - group_pad)
        gx2 = min(original.shape[1], int(group_box["x"] + group_box["width"]) + group_pad)
        gy2 = min(original.shape[0], int(group_box["y"] + group_box["height"]) + group_pad)
        group_crop = original[gy1:gy2, gx1:gx2]
        group_gray = cv2.cvtColor(group_crop, cv2.COLOR_RGB2GRAY)
        group_edge = np.concatenate((group_gray[0], group_gray[-1], group_gray[:, 0], group_gray[:, -1]))
        full_line_ratio = sum(
            detail.get("reason") == "SAFE_FLAT_BACKGROUND_FULL" for detail in details
        ) / max(1, len(details))
        is_bubble = (
            group.get("style", "dialogue") in {"dialogue", "thought", "shout"}
            or (surface or {}).get("surface_type") in {"BUBBLE_FLAT", "BUBBLE_CURVED", "bubble"}
        )
        full_group_safe = bool(
            not is_bubble
            and np.mean(group_gray >= 205) >= 0.60
            and np.mean(group_edge >= 205) >= (0.82 if full_line_ratio >= 0.60 else 0.90)
        )
        if full_group_safe:
            output, painted_mask = reconstruct_flat_surface(output, (gx1, gy1, gx2, gy2))
            allowed_mask = cv2.bitwise_or(allowed_mask, painted_mask)
            results.append({"id": group["id"], "status": "CLEANED", "regions": details, "group_clear": True})
            continue

        # Sample one conservative flat background color per OCR line. Expand a
        # little beyond the tight OCR rectangle so italic overhangs, outlines,
        # and JPEG antialiasing cannot remain beneath the replacement text.
        painted_mask = np.zeros_like(group_mask)
        for detail in details:
            region = by_id[detail["region_id"]]
            line_mask, _ = region_mask(original, region["bbox"])
            line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8), iterations=2)
            color = np.asarray(detail["fill_rgb"], dtype=np.uint8)
            output[line_mask > 0] = color
            painted_mask = cv2.bitwise_or(painted_mask, line_mask)
        allowed_mask = cv2.bitwise_or(allowed_mask, painted_mask)
        results.append({"id": group["id"], "status": "CLEANED", "regions": details,
                        "surface_type": surface.get("surface_type") if surface else None})

    changed = np.any(output != original, axis=2)
    outside_changes = int(np.count_nonzero(changed & (allowed_mask == 0)))
    if outside_changes:
        raise RuntimeError(f"Pixel safety failed: {outside_changes} pixels changed outside mask")
    if output.shape != original.shape:
        raise RuntimeError("Output dimensions differ from original")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output).save(args.output, format="PNG", optimize=True)
    Image.fromarray(allowed_mask).save(args.mask, format="PNG", optimize=True)
    report = {
        "input_dimensions": [original.shape[1], original.shape[0]],
        "output_dimensions": [output.shape[1], output.shape[0]],
        "seconds": round(time.perf_counter() - started, 2),
        "changed_pixels": int(np.count_nonzero(changed)),
        "allowed_mask_pixels": int(np.count_nonzero(allowed_mask)),
        "outside_mask_changed_pixels": outside_changes,
        "results": results,
        "hybrid": {
            "mode": args.hybrid_mode,
            "lama_available": hybrid_runtime_available,
            "inference_enabled": args.hybrid_mode == "on",
            "artwork_events": len(hybrid_events),
            "events": hybrid_events,
        },
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.hybrid_report:
        args.hybrid_report.parent.mkdir(parents=True, exist_ok=True)
        args.hybrid_report.write_text(json.dumps(report["hybrid"], indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
