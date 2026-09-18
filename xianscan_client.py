#!/usr/bin/env python3
"""Small client for the pinned local XianScan ML sidecar."""

from __future__ import annotations

import json
import base64

import cv2
import numpy as np
import requests


def _clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, int(value)))


def normalize_analyze_response(payload: dict, expected_shape: tuple[int, int] | None = None,
                               require_text: bool = True) -> dict:
    """Convert XianScan analyze output into the translator OCR schema."""
    width, height = int(payload.get("width", 0)), int(payload.get("height", 0))
    if width <= 0 or height <= 0:
        raise ValueError("XianScan analyze returned invalid dimensions")
    if expected_shape is not None:
        expected_height, expected_width = expected_shape
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"XianScan analyze changed dimensions: "
                f"{(expected_width, expected_height)} -> {(width, height)}"
            )
    regions = []
    for index, item in enumerate(payload.get("regions") or []):
        box = item.get("box") or {}
        x = _clamp(box.get("x", 0), 0, width - 1)
        y = _clamp(box.get("y", 0), 0, height - 1)
        box_width = _clamp(box.get("w", 0), 0, width - x)
        box_height = _clamp(box.get("h", 0), 0, height - y)
        text = str(item.get("text", "")).strip()
        if (require_text and not text) or box_width <= 0 or box_height <= 0:
            continue
        polygon = []
        for point in item.get("polygon") or []:
            if isinstance(point, (list, tuple)) and len(point) == 2:
                polygon.append([
                    _clamp(point[0], 0, width - 1),
                    _clamp(point[1], 0, height - 1),
                ])
        if len(polygon) < 3:
            polygon = [[x, y], [x + box_width, y],
                       [x + box_width, y + box_height], [x, y + box_height]]
        confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        regions.append({
            "id": f"xianscan:{item.get('id', index)}", "text": text,
            "confidence": confidence,
            "bbox": {"x": x, "y": y, "width": box_width, "height": box_height},
            "polygon": polygon, "type": "dialogue",
            "status": "OK" if confidence >= 0.75 else "LOW_OCR_CONFIDENCE",
            "detector_backend": str(payload.get("backend", "unknown")),
            "vertical": bool(item.get("vertical", False)),
            "angle": float(item.get("angle", 0.0)),
        })
    return {"width": width, "height": height, "regions": regions,
            "region_count": len(regions), "backend": str(payload.get("backend", "unknown"))}


def bbox_iou(left: dict, right: dict) -> float:
    left_x2, left_y2 = left["x"] + left["width"], left["y"] + left["height"]
    right_x2, right_y2 = right["x"] + right["width"], right["y"] + right["height"]
    intersection = max(0, min(left_x2, right_x2) - max(left["x"], right["x"])) * max(
        0, min(left_y2, right_y2) - max(left["y"], right["y"])
    )
    union = left["width"] * left["height"] + right["width"] * right["height"] - intersection
    return intersection / union if union else 0.0


def match_regions(primary: list[dict], secondary: list[dict], minimum_iou: float = 0.20) -> list[dict]:
    """Build a one-to-one geometry comparison without changing OCR output."""
    available = set(range(len(secondary)))
    matches = []
    for left in primary:
        candidates = [(bbox_iou(left["bbox"], secondary[index]["bbox"]), index)
                      for index in available]
        score, selected = max(candidates, default=(0.0, -1))
        if selected >= 0 and score >= minimum_iou:
            available.remove(selected)
            matches.append({"primary_id": left["id"],
                            "secondary_id": secondary[selected]["id"], "iou": round(score, 4)})
        else:
            matches.append({"primary_id": left["id"], "secondary_id": None,
                            "iou": round(score, 4)})
    return matches


def offset_region(region: dict, offset_y: int, page_height: int) -> dict:
    """Move a tile-local region into full-page coordinates."""
    item = {**region, "bbox": dict(region["bbox"]),
            "polygon": [list(point) for point in region.get("polygon", [])]}
    item["bbox"]["y"] = _clamp(item["bbox"]["y"] + offset_y, 0, page_height - 1)
    item["polygon"] = [[point[0], _clamp(point[1] + offset_y, 0, page_height - 1)]
                       for point in item["polygon"]]
    return item


def merge_tile_regions(regions: list[dict]) -> list[dict]:
    """Deduplicate overlapping tile results, preferring confidence."""
    accepted: list[dict] = []
    for candidate in sorted(regions, key=lambda item: (item["bbox"]["y"], item["bbox"]["x"])):
        duplicate_index = None
        normalized = "".join(candidate["text"].casefold().split())
        for index in range(max(0, len(accepted) - 30), len(accepted)):
            existing = accepted[index]
            same_text = "".join(existing["text"].casefold().split()) == normalized
            overlap = bbox_iou(candidate["bbox"], existing["bbox"])
            if overlap >= 0.60 or (same_text and overlap >= 0.20):
                duplicate_index = index
                break
        if duplicate_index is None:
            accepted.append(candidate)
        elif candidate["confidence"] > accepted[duplicate_index]["confidence"]:
            accepted[duplicate_index] = candidate
    accepted.sort(key=lambda item: (item["bbox"]["y"], item["bbox"]["x"]))
    return accepted


class XianScanClient:
    def __init__(self, base_url: str, timeout: int = 180, session=None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self) -> dict:
        response = self.session.get(f"{self.base_url}/health", timeout=10)
        response.raise_for_status()
        return response.json()

    def analyze(self, image_rgb: np.ndarray) -> dict:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode XianScan analyze input")
        response = self.session.post(
            f"{self.base_url}/pages/analyze",
            files={"image": ("page.png", encoded.tobytes(), "image/png")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return normalize_analyze_response(response.json(), image_rgb.shape[:2])

    def detect(self, image_rgb: np.ndarray, include_mask: bool = False) -> dict:
        """Run detector geometry only; no RapidOCR recognition is performed."""
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode XianScan detect input")
        response = self.session.post(
            f"{self.base_url}/pages/detect",
            files={"image": ("page.png", encoded.tobytes(), "image/png")},
            data={"include_mask": "true" if include_mask else "false"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        result = normalize_analyze_response(payload, image_rgb.shape[:2], require_text=False)
        result["text_mask"] = None
        encoded_mask = payload.get("mask_png_base64")
        if include_mask and encoded_mask:
            try:
                raw = base64.b64decode(encoded_mask, validate=True)
                mask = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            except (ValueError, TypeError) as error:
                raise RuntimeError("XianScan returned an invalid detector mask") from error
            if mask is None or mask.shape != image_rgb.shape[:2]:
                raise RuntimeError("XianScan detector mask dimensions differ from image")
            result["text_mask"] = mask
        return result

    def inpaint_mask(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Send one exact page mask to the shared XianScan LaMa process."""
        if mask.shape != image_rgb.shape[:2]:
            raise ValueError("XianScan inpaint mask dimensions differ from image")
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not np.any(binary):
            return image_rgb.copy()

        # The sidecar deliberately rejects masks that cover too much of a page.
        # Split them here, at the common client boundary, so every caller
        # (dialogue cleanup, logos, banners, and branding) gets the same safe
        # behaviour.  Horizontal bands also guarantee that even a completely
        # filled mask stays below the server limit; sparse bands are cheaper.
        max_mask_ratio = 0.15
        if np.count_nonzero(binary) / binary.size > max_mask_ratio:
            height = binary.shape[0]
            band_height = max(1, int(height * max_mask_ratio))
            result = image_rgb.copy()
            for top in range(0, height, band_height):
                bottom = min(height, top + band_height)
                band_mask = np.zeros_like(binary)
                band_mask[top:bottom] = binary[top:bottom]
                if np.any(band_mask):
                    result = self.inpaint_mask(result, band_mask)
            return result

        # Do not upload an entire high-resolution webtoon page for a small
        # text mask. Besides wasting memory, detailed pages can exceed the
        # sidecar upload limit. Give LaMa enough surrounding context, then
        # paste only explicitly masked pixels back into the original page.
        points = cv2.findNonZero(binary)
        x, y, width, height = cv2.boundingRect(points)
        context = max(96, min(256, round(max(width, height) * 0.20)))
        left, top = max(0, x - context), max(0, y - context)
        right = min(binary.shape[1], x + width + context)
        bottom = min(binary.shape[0], y + height + context)
        # Cropping changes the denominator used by the sidecar's mask safety
        # check. Grow the context until the exact same safe page mask remains
        # below that limit in crop coordinates as well.
        page_height, page_width = binary.shape
        while (
            np.count_nonzero(binary[top:bottom, left:right])
            / max(1, (bottom - top) * (right - left))
            > 0.18
            and (left > 0 or top > 0 or right < page_width or bottom < page_height)
        ):
            grow_x = max(96, (right - left) // 2)
            grow_y = max(96, (bottom - top) // 2)
            left, top = max(0, left - grow_x), max(0, top - grow_y)
            right = min(page_width, right + grow_x)
            bottom = min(page_height, bottom + grow_y)
        crop_image = image_rgb[top:bottom, left:right]
        crop_mask = binary[top:bottom, left:right]

        # JPEG keeps very tall, texture-heavy webtoon requests comfortably
        # below the 32 MiB transport cap. The response is composited back only
        # where crop_mask is set, so JPEG differences cannot touch other art.
        ok_image, encoded_image = cv2.imencode(
            ".jpg", cv2.cvtColor(crop_image, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        ok_mask, encoded_mask = cv2.imencode(".png", crop_mask)
        if not ok_image or not ok_mask:
            raise RuntimeError("Could not encode XianScan masked inpaint input")
        response = self.session.post(
            f"{self.base_url}/pages/inpaint-mask",
            files={
                "image": ("page.jpg", encoded_image.tobytes(), "image/jpeg"),
                "mask": ("mask.png", encoded_mask.tobytes(), "image/png"),
            },
            data={"mode": "patch"}, timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as error:
            failed=getattr(error,"response",None) or response
            detail=str(getattr(failed,"text","")).strip()[:500]
            status=getattr(failed,"status_code","unknown")
            reason=getattr(failed,"reason","request rejected")
            raise RuntimeError(
                f"XianScan masked inpaint rejected the request "
                f"({status}): {detail or reason}"
            ) from error
        decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("XianScan returned an invalid masked inpaint image")
        crop_result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if crop_result.shape != crop_image.shape:
            raise RuntimeError("XianScan masked inpaint changed dimensions")
        result = image_rgb.copy()
        target = result[top:bottom, left:right]
        selected = crop_mask > 0
        target[selected] = crop_result[selected]
        # Validate the actual page returned to the caller, not the JPEG
        # transport response (which naturally differs outside the mask).
        changed = np.any(result != image_rgb, axis=2)
        if np.any(changed & (binary == 0)):
            raise RuntimeError("XianScan masked inpaint changed pixels outside mask")
        return result

    def analyze_tiled(self, image_rgb: np.ndarray, tile_height: int = 1600,
                      overlap: int = 160) -> dict:
        """Analyze a long page serially without sending it whole to the sidecar."""
        if tile_height < 256 or overlap < 0 or overlap >= tile_height:
            raise ValueError("Invalid XianScan tile height/overlap")
        height, width = image_rgb.shape[:2]
        regions = []
        backends = set()
        tile_count = 0
        top = 0
        while top < height:
            bottom = min(height, top + tile_height)
            result = self.analyze(image_rgb[top:bottom])
            tile_count += 1
            backends.add(result["backend"])
            regions.extend(offset_region(item, top, height) for item in result["regions"])
            if bottom == height:
                break
            top = bottom - overlap
        merged = merge_tile_regions(regions)
        return {
            "width": width, "height": height, "regions": merged,
            "region_count": len(merged), "tile_count": tile_count,
            "backend": "+".join(sorted(backends)) if backends else "unknown",
        }

    def clean(self, image_rgb: np.ndarray, regions: list[dict]) -> np.ndarray:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("Could not encode XianScan input")
        response = self.session.post(
            f"{self.base_url}/pages/clean",
            files={"image": ("page.png", encoded.tobytes(), "image/png")},
            data={"regions": json.dumps(regions), "inpaint_mode": "patch"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        decoded = cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise RuntimeError("XianScan returned an invalid image")
        result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        if result.shape != image_rgb.shape:
            raise RuntimeError(f"XianScan changed dimensions: {image_rgb.shape} -> {result.shape}")
        return result


def payload_for_regions(regions: list[dict]) -> list[dict]:
    payload = []
    for region in regions:
        box = region["bbox"]
        payload.append({
            "id": str(region["id"]),
            "box": {"x": int(box["x"]), "y": int(box["y"]),
                    "w": int(box["width"]), "h": int(box["height"])},
            "polygon": [[int(point[0]), int(point[1])] for point in region.get("polygon", [])],
        })
    return payload


def allowed_mask(shape: tuple[int, int], regions: list[dict], dilation: int = 4) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for region in regions:
        points = np.asarray(region.get("polygon") or [], dtype=np.int32)
        if len(points) >= 3:
            cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
        else:
            box = region["bbox"]
            x1, y1 = int(box["x"]), int(box["y"])
            x2, y2 = x1 + int(box["width"]), y1 + int(box["height"])
            mask[y1:y2, x1:x2] = 255
    if dilation and np.any(mask):
        mask = cv2.dilate(mask, np.ones((dilation * 2 + 1, dilation * 2 + 1), np.uint8))
    return mask
