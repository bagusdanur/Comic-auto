#!/usr/bin/env python3
"""Replace source scanlation credits with Ryukomik branding after typesetting."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


CREDIT_RE = re.compile(
    r"(?:scan(?:s|lation)?|thunder\s*scans?|en[-\s]*thunder|"
    r"asura\s*(?:scans?|comics?)?|reaper\s*scans?|flame\s*(?:comics?|scans?)?|"
    r"void\s*scans?|night\s*scans?|aurora\s*scans?|quantum\s*scans?|"
    r"manhwafreak|freak\s*scans?|cosmic\s*scans?|drake\s*scans?|"
    r"luminous\s*scans?|realm\s*scans?|reset\s*scans?|surreal\s*scans?|"
    r"anigli\s*scans?|eva\s*scans?|evascans\.org|topmanhua|manganato|mangakakalot|bato\.to|kun\s*manga|"
    r"read\s+(?:this|only|at|on)|(?:read|reading)\b.{0,28}\b(?:site|website)|"
    r"read\b.{0,24}\b(?:chapter|series)\b.{0,16}\b(?:at|on)|"
    r"(?:please\s+)?support\s+(?:us|our)|support\s+means|"
    r"reader\b.{0,36}\b(?:support|website)|more\s+amazing\s+series|"
    r"discover\s+more\s+chapters?|"
    r"best\s+possible\s+quality|continuing\s+to\s+publish|"
    r"only\s+at|website|discord|paypal|"
    r"patreon|ko-?fi|t\.me|https?://|www\.|\.(?:com|org|net|gg|to)\b|"
    r"recruit(?:ment|ing)?|proofreaders?|(?:korean|japanese)\s+translators?|"
    r"join\b.{0,30}\bdiscord)",
    re.IGNORECASE,
)


def credit_regions(ocr: dict) -> list[dict]:
    return [region for region in ocr.get("regions", []) if CREDIT_RE.search(region.get("text", ""))]


def visual_credit_regions(image: Image.Image, existing: list[dict]) -> list[dict]:
    """Find small dark scanlator pills on light gaps when OCR misses their text."""
    try:
        import cv2
    except ImportError:
        return []
    array = np.asarray(image.convert("RGB"))
    height, width = array.shape[:2]
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    dark = ((gray < 105).astype(np.uint8) * 255)
    dark[:, round(width * 0.90):] = 0
    joined = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 17), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(joined)
    found: list[dict] = []
    for x, y, item_width, item_height, area in stats[1:count]:
        if not (width * 0.12 <= item_width <= width * 0.65):
            continue
        if not (width * 0.040 <= item_height <= width * 0.35):
            continue
        if not (1.4 <= item_width / max(item_height, 1) <= 6.5):
            continue
        crop = gray[y:y + item_height, x:x + item_width]
        if float(np.mean(crop < 115)) < 0.24:
            continue
        pad = max(8, round(width * 0.025))
        context = gray[max(0, y - pad):min(height, y + item_height + pad),
                       max(0, x - pad):min(width, x + item_width + pad)]
        if float(np.mean(context >= 220)) < 0.28:
            continue
        candidate = {"text": "__visual_scan_logo__", "bbox": {
            "x": int(x), "y": int(y), "width": int(item_width), "height": int(item_height),
        }}
        center_y = y + item_height / 2
        if any(abs(center_y - (item["bbox"]["y"] + item["bbox"]["height"] / 2)) < item_height for item in existing):
            continue
        found.append(candidate)
    return found


def clusters(regions: list[dict], max_gap: int) -> list[list[dict]]:
    ordered = sorted(regions, key=lambda item: item["bbox"]["y"])
    groups: list[list[dict]] = []
    for region in ordered:
        y = int(region["bbox"]["y"])
        if not groups:
            groups.append([region])
            continue
        previous = groups[-1][-1]["bbox"]
        previous_bottom = int(previous["y"] + previous["height"])
        if y - previous_bottom <= max_gap:
            groups[-1].append(region)
        else:
            groups.append([region])
    return groups


def resized(asset: Image.Image, width: int, opacity: float = 1.0) -> Image.Image:
    width = max(1, width)
    height = max(1, round(asset.height * width / asset.width))
    result = asset.resize((width, height), Image.Resampling.LANCZOS)
    if opacity < 1:
        alpha = result.getchannel("A").point(lambda value: round(value * opacity))
        result.putalpha(alpha)
    return result


def alpha_composite_clipped(base: Image.Image, overlay: Image.Image, x: int, y: int) -> None:
    """Alpha-composite an overlay even when its centered position crosses an edge."""
    left, top = max(0, -x), max(0, -y)
    right = min(overlay.width, base.width - x)
    bottom = min(overlay.height, base.height - y)
    if right <= left or bottom <= top:
        return
    base.alpha_composite(overlay.crop((left, top, right, bottom)), (max(0, x), max(0, y)))


def inpaint_old_mark(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    """Remove the old mark locally so it cannot show through transparent logo gaps."""
    try:
        import cv2
    except ImportError:
        return image
    array = np.asarray(image.convert("RGB")).copy()
    x1, y1, x2, y2 = box
    mask = np.zeros(array.shape[:2], dtype=np.uint8)
    # Rounded masks avoid leaving the source avatar corners while producing a
    # softer reconstruction boundary than a hard white/background rectangle.
    radius = max(5, min((x2 - x1) // 5, (y2 - y1) // 2))
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1)
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1)
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)
    restored = cv2.inpaint(array, mask, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(restored).convert("RGBA")


def local_background(array: np.ndarray, y: int, height: int = 24) -> tuple[int, int, int]:
    y1 = min(array.shape[0] - 1, max(0, y))
    y2 = min(array.shape[0], y1 + height)
    strip = array[y1:y2]
    if not strip.size:
        return (255, 255, 255)
    return tuple(np.median(strip.reshape(-1, 3), axis=0).round().astype(np.uint8))


def find_top_credit_end(array: np.ndarray, minimum_y: int) -> int | None:
    """Find the first clean, nearly uniform full-width band below a top credit card."""
    height, width = array.shape[:2]
    start = min(height - 1, max(minimum_y, 1))
    band_height = max(20, round(width * 0.055))
    for y in range(start, min(height - band_height, round(height * 0.35)), 4):
        band = array[y:y + band_height]
        gray = np.mean(band, axis=2)
        if np.mean(gray >= 235) >= 0.965 and float(np.std(band)) <= 16:
            return y
    return None


def clear_top_credit(image: Image.Image, regions: list[dict]) -> tuple[Image.Image, int | None]:
    array = np.asarray(image.convert("RGB"))
    top = [item for item in regions if item["bbox"]["y"] < image.height * 0.18]
    if len(top) < 2:
        return image, None
    minimum = max(int(item["bbox"]["y"] + item["bbox"]["height"]) for item in top)
    end = find_top_credit_end(array, minimum)
    if end is None:
        intro_text = " ".join(item.get("text", "") for item in top)
        promo_intro = bool(re.search(
            r"(?:reader\b.{0,36}\b(?:support|website)|more\s+amazing\s+series|"
            r"best\s+possible\s+quality|continuing\s+to\s+publish)", intro_text, re.I,
        ))
        if promo_intro and len(top) >= 3:
            end = min(round(image.height * 0.20), minimum + round(image.width * 0.035))
    if end is None:
        return image, None
    color = local_background(array, end, max(20, round(image.width * 0.05)))
    result = image.copy()
    result.paste(color, (0, 0, image.width, end))
    return result, end


def trim_source_header(image: Image.Image, regions: list[dict]) -> tuple[Image.Image, int | None]:
    """Crop a complete scanlator promo card prepended to the first comic page."""
    top_limit = image.height * 0.10
    anchors = [item for item in regions if int(item["bbox"]["y"]) < top_limit]
    text = " ".join(str(item.get("text", "")) for item in anchors)
    promo_signals = len(re.findall(
        r"(?:qiscans?|oi\s*scans?|discord|recruit|support|chapters?\s+up\s+to|"
        r"available\s+exclusively|follow\s+for|faster\s+updates?)", text, re.I,
    ))
    thunder_warning_card = bool(
        re.search(r"thunder\s*scans?|en[-\s]*thunder", text, re.I)
        and re.search(r"read\s+at|please\s+read|support\s+us|continuing\s+to\s+publish", text, re.I)
        and len(anchors) >= 3
    )
    if (len(anchors) < 4 and not thunder_warning_card) or (promo_signals < 2 and not thunder_warning_card):
        return image, None
    bottom = max(int(item["bbox"]["y"] + item["bbox"]["height"]) for item in anchors)
    boundary = min(round(image.height * 0.14), bottom + round(image.width * 0.055))
    if boundary < image.width * 0.45:
        return image, None
    return image.crop((0, boundary, image.width, image.height)), boundary


def boxes_overlap(a: tuple[int, int, int, int], box: dict, padding: int = 0) -> bool:
    bx1 = int(box["x"]) - padding
    by1 = int(box["y"]) - padding
    bx2 = int(box["x"] + box["width"]) + padding
    by2 = int(box["y"] + box["height"]) + padding
    return a[0] < bx2 and a[2] > bx1 and a[1] < by2 and a[3] > by1


def substantially_same_area(a: dict, b: dict) -> bool:
    """Treat duplicate OCR boxes over a detected credit as the same mark."""
    ax1, ay1 = int(a["x"]), int(a["y"])
    ax2, ay2 = ax1 + int(a["width"]), ay1 + int(a["height"])
    bx1, by1 = int(b["x"]), int(b["y"])
    bx2, by2 = bx1 + int(b["width"]), by1 + int(b["height"])
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    smaller = max(1, min((ax2 - ax1) * (ay2 - ay1), (bx2 - bx1) * (by2 - by1)))
    return overlap / smaller >= 0.45


def replace_credit_cluster(
    image: Image.Image, group: list[dict], logo: Image.Image, occupied: list[dict] | None = None,
) -> dict | None:
    boxes = [item["bbox"] for item in group]
    group_text = " ".join(item.get("text", "") for item in group)
    is_thunderscans = bool(re.search(r"(?:thunder\s*scans?|en[-\s]*thunder)", group_text, re.I))
    # A plain READ AT domain line should disappear. A designed scanlator pill
    # ("you can read the chapter on") is a logo and must be overlaid directly.
    remove_only = is_thunderscans and bool(re.search(r"\bread\s+at\b", group_text, re.I))
    direct_overlay = is_thunderscans and not remove_only
    confirmed_scanlator = is_thunderscans or any(
        re.search(r"\.com\b", item.get("text", ""), re.I) for item in group
    )
    width, height = image.size
    protected = occupied or []
    collision_padding = max(8, round(width * 0.018))

    def safe_logo_position(mark: Image.Image, preferred: tuple[int, int, int, int]) -> tuple[int, int] | None:
        """Keep replacement branding away from dialogue, even for confirmed credits."""
        left, top, right, bottom = preferred
        candidates = [
            (left + ((right - left) - mark.width) // 2,
             top + ((bottom - top) - mark.height) // 2),
            (left, top - mark.height - collision_padding),
            (left, bottom + collision_padding),
            (right + collision_padding, top),
            (left - mark.width - collision_padding, top),
        ]
        # Stable corner fallbacks are useful when the source mark itself was
        # printed across a speech bubble. They are deliberately inset.
        inset = max(12, round(width * 0.025))
        candidates.extend([
            (inset, inset), (width - mark.width - inset, inset),
            (inset, height - mark.height - inset),
            (width - mark.width - inset, height - mark.height - inset),
        ])
        seen: set[tuple[int, int]] = set()
        for raw_x, raw_y in candidates:
            px = max(0, min(width - mark.width, raw_x))
            py = max(0, min(height - mark.height, raw_y))
            if (px, py) in seen:
                continue
            seen.add((px, py))
            logo_box = (px, py, px + mark.width, py + mark.height)
            if not any(boxes_overlap(logo_box, item["bbox"], collision_padding) for item in protected):
                return px, py
        return None
    if remove_only:
        pad = max(7, round(width * 0.012))
        if any(boxes_overlap((int(b['x'])-pad,int(b['y'])-pad,int(b['x']+b['width'])+pad,int(b['y']+b['height'])+pad), item['bbox'], collision_padding) for b in boxes for item in protected):
            return None
        # Plain credit text is erased line-by-line. A single expanded inpaint
        # box can smear nearby artwork (for example the phone below READ AT).
        covered: list[list[int]] = []
        for box in boxes:
            # Stylized italic credit fonts extend beyond PaddleOCR's tight box.
            # A modest per-line pad removes those tips without reaching the
            # neighboring artwork as the previous group-wide erase did.
            pad = max(7, round(width * 0.012))
            erase_box = (
                max(0, int(box["x"]) - pad), max(0, int(box["y"]) - pad),
                min(width, int(box["x"] + box["width"]) + pad),
                min(height, int(box["y"] + box["height"]) + pad),
            )
            crop = np.asarray(image.convert("RGB"))[erase_box[1]:erase_box[3], erase_box[0]:erase_box[2]]
            if crop.size and float(np.mean(np.mean(crop, axis=2) >= 225)) >= 0.58:
                image.paste((255, 255, 255, 255), erase_box)
            else:
                restored = inpaint_old_mark(image, erase_box)
                image.paste(restored)
            covered.append(list(erase_box))
        union = [min(b[0] for b in covered), min(b[1] for b in covered),
                 max(b[2] for b in covered), max(b[3] for b in covered)]
        return {"covered": union, "covered_lines": covered, "logo": None, "mode": "removed_source_credit"}

    if direct_overlay:
        min_x = min(int(box["x"]) for box in boxes)
        min_y = min(int(box["y"]) for box in boxes)
        max_x = max(int(box["x"] + box["width"]) for box in boxes)
        max_y = max(int(box["y"] + box["height"]) for box in boxes)
        group_height = max(1, max_y - min_y)
        erase_box = (
            max(0, min_x - max(round(group_height * 2.8), round(width * 0.055))),
            max(0, min_y - max(round(group_height * 0.75), round(width * 0.012))),
            min(width, max_x + max(5, round(width * 0.010))),
            min(height, max_y + max(round(group_height * 0.75), round(width * 0.012))),
        )
        if any(boxes_overlap(erase_box, item['bbox'], collision_padding) for item in protected):
            return None
        restored = inpaint_old_mark(image, erase_box)
        image.paste(restored)
        target_width = min(round(width * 0.20), max(round(width * 0.12), round((erase_box[2] - erase_box[0]) * 0.72)))
        mark = resized(logo, target_width)
        placement = safe_logo_position(mark, erase_box)
        if placement is None:
            return {"covered": list(erase_box), "logo": None, "mode": "removed_source_credit_no_safe_logo"}
        px, py = placement
        alpha_composite_clipped(image, mark, px, py)
        return {"covered": list(erase_box), "logo": [px, py, mark.width, mark.height], "mode": "direct_logo_overlay"}
    margin_x = max(20, round(width * 0.060))
    margin_y = max(18, round(width * 0.060))
    x1 = max(0, min(int(box["x"]) for box in boxes) - margin_x)
    y1 = max(0, min(int(box["y"]) for box in boxes) - margin_y)
    x2 = min(width, max(int(box["x"] + box["width"]) for box in boxes) + margin_x)
    y2 = min(height, max(int(box["y"] + box["height"]) for box in boxes) + margin_y)
    # Scanlation pills commonly put a round avatar just left of the OCR text.
    if x1 < width * 0.18:
        x1 = 0

    if any(boxes_overlap((x1,y1,x2,y2), item['bbox'], collision_padding) for item in protected):
        return None

    target_width = min(round(width * 0.20), max(round(width * 0.12), round((x2 - x1) * 0.82)))
    mark = resized(logo, target_width)
    placement = safe_logo_position(mark, (x1, y1, x2, y2))
    if placement is None:
        restored = inpaint_old_mark(image, (x1, y1, x2, y2))
        image.paste(restored)
        return {"covered": [x1, y1, x2, y2], "logo": None, "mode": "removed_source_credit_no_safe_logo"}
    px, py = placement
    logo_box = (px, py, px + mark.width, py + mark.height)
    padding = max(8, round(width * 0.018))
    if any(boxes_overlap(logo_box, item["bbox"], padding) for item in protected):
        # A long domain credit is often placed tightly between real panels.
        # Fall back to its exact OCR footprint instead of leaving the raw logo
        # behind or enlarging Ryukomik over neighboring dialogue.
        if not confirmed_scanlator and any(
            boxes_overlap(
                (int(box["x"]), int(box["y"]), int(box["x"] + box["width"]), int(box["y"] + box["height"])),
                item["bbox"], 0,
            )
            for box in boxes for item in protected
        ):
            return None
        tight_pad = max(2, round(width * 0.006))
        min_box_x = min(int(box["x"]) for box in boxes)
        group_height = max(int(box["height"]) for box in boxes)
        # OCR usually sees only the domain inside a scanlator pill and misses
        # its circular avatar/icon on the left. Include that icon so no purple
        # or colored crescent remains visible beside the Ryukomik logo.
        left_avatar = max(tight_pad, round(group_height * 2.6), round(width * 0.045))
        x1 = max(0, min_box_x - left_avatar)
        y1 = max(0, min(int(box["y"]) for box in boxes) - tight_pad)
        x2 = min(width, max(int(box["x"] + box["width"]) for box in boxes) + tight_pad)
        y2 = min(height, max(int(box["y"] + box["height"]) for box in boxes) + tight_pad)
        # Preserve the actual Ryukomik logo proportions. Do not squash it into
        # a scanlator's very short pill/banner, which makes it look like a tiny
        # unreadable strip. Its width follows the source mark and its natural
        # height may extend above/below the old pill.
        fit_width = min(round(width * 0.20), max(round(width * 0.12), round((x2 - x1) * 0.82)))
        mark = resized(logo, fit_width)
        placement = safe_logo_position(mark, (x1, y1, x2, y2))
        if placement is None:
            restored = inpaint_old_mark(image, (x1, y1, x2, y2))
            image.paste(restored)
            return {"covered": [x1, y1, x2, y2], "logo": None, "mode": "removed_source_credit_no_safe_logo"}
        px, py = placement
        logo_box = (px, py, px + mark.width, py + mark.height)
        if any(boxes_overlap(logo_box, item["bbox"], collision_padding) for item in protected):
            return None

    # Only erase and replace after proving the final logo cannot cover dialogue.
    if not direct_overlay:
        restored = inpaint_old_mark(image, (x1, y1, x2, y2))
        image.paste(restored)
    alpha_composite_clipped(image, mark, px, py)
    return {
        "covered": [x1, y1, x2, y2], "logo": [px, py, mark.width, mark.height],
        "mode": "direct_logo_overlay" if direct_overlay else "replaced_source_credit",
    }


def trim_source_footer(image: Image.Image, regions: list[dict]) -> tuple[Image.Image, int | None]:
    """Remove a source scanlator promo card occupying the very end of a chapter."""
    strong = re.compile(
        r"(?:scan|discord|paypal|patreon|ko-?fi|https?://|www\.|"
        r"\.(?:com|org|net|gg)\b|recruit(?:ment|ing)?|proofreaders?|"
        r"(?:korean|japanese)\s+translators?)", re.I,
    )
    # Prefer the explicit start of a recruitment card. Earlier scanlator logos
    # can sit inside the final comic panel several thousand pixels above the
    # appended ad; including those in one generic credit union crops real art.
    recruitment_start = re.compile(
        r"(?:\bjoin\b.{0,35}\bdiscord\b|\brecruit(?:ment|ing)?\b)", re.I,
    )
    recruitment_anchors = [
        item for item in regions
        if not str(item.get("text", "")).startswith("__")
        and recruitment_start.search(item.get("text", ""))
        and int(item["bbox"]["y"]) >= image.height * 0.50
    ]
    if recruitment_anchors:
        first_y = min(int(item["bbox"]["y"]) for item in recruitment_anchors)
        boundary = max(1, first_y - round(image.width * 0.10))
        return image.crop((0, 0, image.width, boundary)), boundary
    # Some sources append a full-width promotional card as the final half-page.
    # OCR may only recognize its giant scanlator name, well above the usual
    # near-bottom threshold. Remove the entire foreign card at its top edge;
    # the Ryukomik footer is appended afterward by the normal branding path.
    large_cards = [
        item for item in regions
        if not str(item.get("text", "")).startswith("__")
        and strong.search(item.get("text", ""))
        and int(item["bbox"]["width"]) >= image.width * 0.65
        and int(item["bbox"]["height"]) >= image.height * 0.18
        and int(item["bbox"]["y"]) >= image.height * 0.25
    ]
    if large_cards:
        first = min(large_cards, key=lambda item: int(item["bbox"]["y"]))
        boundary = max(1, int(first["bbox"]["y"]) - round(image.width * 0.10))
        return image.crop((0, 0, image.width, boundary)), boundary
    promo_regions = [
        item for item in regions
        if not str(item.get("text", "")).startswith("__")
        and strong.search(item.get("text", ""))
        and int(item["bbox"]["y"]) >= image.height * 0.25
    ]
    if len(promo_regions) >= 2:
        union_x1 = min(int(item["bbox"]["x"]) for item in promo_regions)
        union_y1 = min(int(item["bbox"]["y"]) for item in promo_regions)
        union_x2 = max(int(item["bbox"]["x"] + item["bbox"]["width"]) for item in promo_regions)
        union_y2 = max(int(item["bbox"]["y"] + item["bbox"]["height"]) for item in promo_regions)
        if union_x2 - union_x1 >= image.width * 0.55 and union_y2 - union_y1 >= image.height * 0.18:
            boundary = max(1, union_y1 - round(image.width * 0.10))
            return image.crop((0, 0, image.width, boundary)), boundary
    near_bottom = [
        item for item in regions
        if item["bbox"]["y"] >= image.height * 0.82 and strong.search(item.get("text", ""))
    ]
    if not near_bottom:
        return image, None
    first_credit_y = min(int(item["bbox"]["y"]) for item in near_bottom)
    estimate = first_credit_y - round(image.width * 0.38)
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    row_colors = np.mean(array, axis=1)
    differences = np.mean(np.abs(np.diff(row_colors, axis=0)), axis=1)
    radius = round(image.width * 0.18)
    start = max(round(image.height * 0.75), estimate - radius)
    end = min(image.height - 1, estimate + radius)
    boundary = start + int(np.argmax(differences[start:end])) if end > start else estimate
    # A true appended promo card has a strong full-width color boundary. When
    # uncertain, preserve the complete ending and append our banner after it.
    trim_y = boundary + 1 if differences[boundary] >= 24 else None
    if trim_y is None:
        return image, None
    return image.crop((0, 0, image.width, trim_y)), trim_y


def empty_white_position(image: Image.Image, mark: Image.Image, occupied: list[dict]) -> tuple[int, int] | None:
    array = np.asarray(image.convert("RGB"))
    width, height = image.size
    pad = max(10, round(width * 0.025))
    candidate_width = min(round(width * 0.56), width - pad * 2)
    candidate = resized(mark, candidate_width, opacity=0.62)
    candidate_height = candidate.height
    if candidate_height >= height:
        return None
    boxes = [item["bbox"] for item in occupied]
    step = max(12, candidate_height // 5)
    x = (width - candidate.width) // 2
    best: tuple[float, int] | None = None
    for y in range(pad, height - candidate_height - pad, step):
        if any(
            x < box["x"] + box["width"] and x + candidate.width > box["x"]
            and y < box["y"] + box["height"] and y + candidate_height > box["y"]
            for box in boxes
        ):
            continue
        crop = array[y:y + candidate_height, x:x + candidate.width]
        gray = np.mean(crop, axis=2)
        white_ratio = float(np.mean(gray >= 242))
        score = white_ratio - min(float(np.std(crop)) / 255, 0.5)
        if white_ratio >= 0.92 and (best is None or score > best[0]):
            best = (score, y)
    return (x, best[1]) if best else None


def apply_branding_legacy(
    image_path: Path,
    ocr_path: Path,
    logo_path: Path,
    text_path: Path,
    *,
    first_page: bool = False,
    banner_path: Path | None = None,
) -> dict:
    with Image.open(image_path) as opened:
        image = opened.convert("RGBA")
    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
    dialogue_regions: list[dict] = []
    translated_credit_regions: list[dict] = []
    translations_path = ocr_path.with_name("translations.json")
    if translations_path.exists():
        try:
            translations = json.loads(translations_path.read_text(encoding="utf-8"))
            for item in translations.get("groups", []):
                if not item.get("bbox"):
                    continue
                source_text = str(item.get("text", ""))
                if re.search(
                    r"(?:SCAN(?:S|LATION)?|THUNDER|READ\s+AT|\.COM\b|DISCORD|"
                    r"RECRUIT(?:MENT|ING)?|PROOFREADERS?|TRANSLATORS?|JOIN\b.{0,30}\bDISCORD)",
                    source_text, re.I,
                ):
                    translated_credit_regions.append({"text": source_text, "bbox": item["bbox"]})
                else:
                    dialogue_regions.append({"text": item.get("translation", ""), "bbox": item["bbox"]})
        except (OSError, ValueError, KeyError, TypeError):
            dialogue_regions = []
    credits = credit_regions(ocr)
    credits.extend(translated_credit_regions)
    # Do not infer logos from generic high-contrast artwork. That heuristic can
    # mistake sound effects, faces, or speech-bubble decoration for scanlator
    # marks and then place branding across real dialogue. OCR-confirmed credit
    # text is conservative and sufficient for automatic production runs.
    report: dict = {
        "source_header_trimmed_at": None, "top_credit_cleared": None,
        "replacements": [], "source_footer_trimmed_at": None,
        "text_watermark": None, "banner": None, "detected_credit_texts": [],
    }
    report["detected_credit_texts"] = [item.get("text", "") for item in credits]

    if first_page:
        image, report["source_header_trimmed_at"] = trim_source_header(image, credits)
        if report["source_header_trimmed_at"] is None:
            image, report["top_credit_cleared"] = clear_top_credit(image, credits)
    header_trim = report["source_header_trimmed_at"]
    working_regions = list(ocr.get("regions", []))
    if header_trim is not None:
        def shift_after_header(items: list[dict]) -> list[dict]:
            shifted = []
            for item in items:
                box = item["bbox"]
                if int(box["y"] + box["height"]) <= header_trim:
                    continue
                clone = dict(item)
                clone["bbox"] = dict(box)
                clone["bbox"]["y"] = max(0, int(box["y"]) - header_trim)
                shifted.append(clone)
            return shifted
        credits = shift_after_header(credits)
        dialogue_regions = shift_after_header(dialogue_regions)
        working_regions = shift_after_header(working_regions)
    # A source promo may share the penultimate image with real dialogue, so it
    # must be trimmed on every page rather than only when our final banner is
    # appended. Conservative footer detection keeps ordinary comic panels.
    if banner_path is not None:
        image, report["source_footer_trimmed_at"] = trim_source_footer(image, credits)
    with Image.open(logo_path) as opened:
        logo = opened.convert("RGBA")
        content_bbox = logo.getchannel("A").getbbox()
        if content_bbox:
            logo = logo.crop(content_bbox)
    # PaddleOCR can emit duplicate/nested boxes for the same logo. A duplicate
    # whose text was read poorly must not be mistaken for dialogue and veto the
    # replacement of an otherwise confirmed scanlator credit.
    protected_regions = [
        item for item in working_regions
        if item not in credits
        and not any(substantially_same_area(item["bbox"], credit["bbox"]) for credit in credits)
    ]
    protected_regions.extend(dialogue_regions)
    for group in clusters(credits, max(80, round(image.width * 0.16))):
        if report["top_credit_cleared"] and min(item["bbox"]["y"] for item in group) < report["top_credit_cleared"]:
            continue
        if report["source_footer_trimmed_at"] and min(item["bbox"]["y"] for item in group) >= report["source_footer_trimmed_at"]:
            continue
        replaced = replace_credit_cluster(image, group, logo, protected_regions)
        if replaced:
            report["replacements"].append(replaced)

    with Image.open(text_path) as opened:
        watermark_text = opened.convert("RGBA")
    position = empty_white_position(image, watermark_text, ocr.get("regions", []) + dialogue_regions)
    if position:
        mark = resized(watermark_text, round(image.width * 0.56), opacity=1.0)
        image.alpha_composite(mark, position)
        report["text_watermark"] = [position[0], position[1], mark.width, mark.height]

    if banner_path:
        with Image.open(banner_path) as opened:
            banner = resized(opened.convert("RGBA"), image.width)
        background_color = local_background(np.asarray(image.convert("RGB")), image.height - 20)
        background = Image.new("RGBA", (image.width, banner.height), background_color + (255,))
        background.alpha_composite(banner, (0, 0))
        extended = Image.new("RGBA", (image.width, image.height + banner.height))
        extended.alpha_composite(image, (0, 0))
        extended.alpha_composite(background, (0, image.height))
        image = extended
        report["banner"] = [0, image.height - banner.height, banner.width, banner.height]

    image.convert("RGB").save(image_path, format="PNG", optimize=True)
    return report


def apply_branding(*args, **kwargs) -> dict:
    from branding_v53 import apply_branding_v53
    return apply_branding_v53(*args, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("ocr", type=Path)
    parser.add_argument("--logo", type=Path, default=Path("assets/branding/watermark.png"))
    parser.add_argument("--watermark-text", type=Path, default=Path("assets/branding/watermark-text.png"))
    parser.add_argument("--first-page", action="store_true")
    parser.add_argument("--banner", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = apply_branding(
        args.image, args.ocr, args.logo, args.watermark_text,
        first_page=args.first_page, banner_path=args.banner,
    )
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
