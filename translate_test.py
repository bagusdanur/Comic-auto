#!/usr/bin/env python3
"""Phase 1 CPU-only OCR benchmark for long English comic images."""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np
import psutil
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageOps

# 1600 instead of 2000: PaddleOCR's text detector merges adjacent lines into
# one box once a tile gets tall, so a mixed Korean/English balloon collapsed
# into a single unreadable box ('y ') and was then dropped by the filters --
# leaving the raw source untouched under the translation. At <=1600 the same
# balloon resolves into one box per line and every line is cleaned.
TILE_HEIGHT = int(os.getenv("TRANSLATOR_TILE_HEIGHT", "1600"))
TILE_OVERLAP = int(os.getenv("TRANSLATOR_TILE_OVERLAP", "200"))
MIN_FREE_RAM_MB = int(os.getenv("TRANSLATOR_MIN_FREE_RAM_MB", "2000"))
CRITICAL_FREE_RAM_MB = int(os.getenv("TRANSLATOR_CRITICAL_FREE_RAM_MB", "512"))
RESOURCE_WAIT_SECONDS = int(os.getenv("TRANSLATOR_RESOURCE_WAIT_SECONDS", "90"))
MAX_CPU_THREADS = int(os.getenv("TRANSLATOR_MAX_CPU_THREADS", "2"))
CONFIDENCE_THRESHOLD = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.75"))

# Lines below the grouping confidence floor are exactly the ones the filters
# throw away, so they get a second reading from a tight crop before that
# happens. Matches the floor: a 0.75 line is already trusted as-is.
RETRY_CONFIDENCE_BELOW = float(os.getenv("OCR_RETRY_CONFIDENCE_BELOW", "0.75"))


def retry_better(joined_letters, joined_score, region_letters, region_score):
    """Accept a tighter re-read in place of the garbage the tile produced.

    Confidence is the judge: a 0.98 reading beats a 0.25 one whatever its
    length.  The length term only stops a narrow crop from collapsing the
    line into a token, so it must be *at least half* the original rather than
    *more* than it.  The old `> region_letters` gate rejected perfect reads
    because the garbage happened to be letter-rich -- the tile rendered
    "MOVING NEXT WEEK, RIGHT?" as 'TXENSNIAOE' (10 letters) and 'SLRIIETEEE'
    (10), while the correct readings are 'MOVING NEXT' (10) and 'WEEK, RIGHT?'
    (9), so `10 > 10` and `9 > 10` both failed at conf 0.98/0.97 and the raw
    English stayed on the finished page.
    """
    if joined_score <= region_score:
        return False
    return joined_letters >= max(3, region_letters // 2)


def available_ram_mb() -> int:
    return int(psutil.virtual_memory().available / 1024 / 1024)


class ResourceSampler:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.peak_rss_mb = 0.0
        self.min_available_mb = available_ram_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.1):
            self.peak_rss_mb = max(
                self.peak_rss_mb, self.process.memory_info().rss / 1024 / 1024
            )
            self.min_available_mb = min(self.min_available_mb, available_ram_mb())
            # The normal queue threshold is a scheduling hint, not a safe
            # reason to kill an OCR process that is already making progress.
            # Only abort at the separate critical floor (normally 512 MB).
            if self.min_available_mb < CRITICAL_FREE_RAM_MB:
                print(
                    f"RESOURCE_GUARD_ABORT: available RAM {self.min_available_mb} MB",
                    flush=True,
                )
                os._exit(3)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()
        self._run()


def tile_ranges(height: int) -> list[tuple[int, int]]:
    if TILE_HEIGHT < 1 or TILE_OVERLAP < 0 or TILE_OVERLAP >= TILE_HEIGHT:
        raise ValueError("Invalid tile height/overlap configuration")
    ranges = []
    top = 0
    while top < height:
        bottom = min(height, top + TILE_HEIGHT)
        ranges.append((top, bottom))
        if bottom == height:
            break
        top = bottom - TILE_OVERLAP
    # A tail tile shorter than half a tile is a hole, not a tile: at
    # TILE_HEIGHT=1600/OVERLAP=200 a 13000 px page ends with (12600,13000),
    # and the detector returns nothing at all from those 400 px while the
    # same pixels read back at 0.80-0.88 confidence from a 1000 px crop.
    # ch1 page-001 puts its 'FEAR' box there (y12815-12999), so it was never
    # detected -- never translated, never removed -- while every other
    # geometry I tried found it. Fold the tail into the tile before it: the
    # merged span covers exactly the same rows, just with enough height for
    # the detector to work in.
    if len(ranges) > 1 and (ranges[-1][1] - ranges[-1][0]) < max(1, TILE_HEIGHT // 2):
        ranges[-2] = (ranges[-2][0], ranges[-1][1])
        ranges.pop()
    return ranges


def bbox_from_polygon(polygon: list[list[int]]) -> dict[str, int]:
    xs = [int(point[0]) for point in polygon]
    ys = [int(point[1]) for point in polygon]
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def intersection_over_union(a: dict[str, int], b: dict[str, int]) -> float:
    left = max(a["x"], b["x"])
    top = max(a["y"], b["y"])
    right = min(a["x"] + a["width"], b["x"] + b["width"])
    bottom = min(a["y"] + a["height"], b["y"] + b["height"])
    intersection = max(0, right - left) * max(0, bottom - top)
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.0


def is_duplicate(candidate: dict, accepted: list[dict]) -> bool:
    normalized = "".join(candidate["text"].casefold().split())
    for existing in reversed(accepted[-20:]):
        if existing["bbox"]["y"] + existing["bbox"]["height"] < candidate["bbox"]["y"] - TILE_OVERLAP:
            break
        same_text = "".join(existing["text"].casefold().split()) == normalized
        if same_text and intersection_over_union(candidate["bbox"], existing["bbox"]) >= 0.25:
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()

    free_before = available_ram_mb()
    wait_deadline = time.monotonic() + RESOURCE_WAIT_SECONDS
    while free_before < MIN_FREE_RAM_MB and free_before >= CRITICAL_FREE_RAM_MB and time.monotonic() < wait_deadline:
        print(f"WAITING_RESOURCE: available RAM {free_before} MB < soft limit {MIN_FREE_RAM_MB} MB", flush=True)
        gc.collect()
        time.sleep(5)
        free_before = available_ram_mb()
    if free_before < CRITICAL_FREE_RAM_MB:
        print(f"RESOURCE_GUARD_ABORT: available RAM {free_before} MB < critical limit {CRITICAL_FREE_RAM_MB} MB")
        return 3
    if free_before < MIN_FREE_RAM_MB:
        print(f"RESOURCE_GUARD_CONTINUE: available RAM {free_before} MB; continuing above critical floor", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sampler = ResourceSampler()
    sampler.start()
    started = time.perf_counter()

    with Image.open(args.input) as opened:
        original_format = opened.format
        original_size = opened.size
        image = ImageOps.exif_transpose(opened).convert("RGB")

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        # en_PP-OCRv5_mobile_rec can only read Latin, and this raw is a Korean
        # manhwa with the English overlay drawn on top of the hangul. The
        # English-only head hallucinated Latin fragments from the mixed line
        # ('y ' for "WHY DID SHE STOP MID-SENTENCE?"), that region then failed
        # the >=3-letter filter and was never cleaned. PP-OCRv5_server_rec is
        # multilingual, reads both scripts, and costs ~15% more OCR time.
        text_recognition_model_name="PP-OCRv5_server_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        # Without the textline angle head the detector still finds a vertical
        # speech line, but the recogniser reads the unrotated strip and returns
        # garbage ('HUY M MAE' conf 0.33 for "JUST WAIT A LITTLE LONGER"),
        # which then fails the confidence filter and is never translated or
        # cleaned. The angle head costs one small inference per line and turns
        # the same regions into 0.9+ confidence reads.
        use_textline_orientation=True,
        device="cpu",
        cpu_threads=MAX_CPU_THREADS,
        enable_mkldnn=True,
    )

    regions: list[dict] = []
    ranges = tile_ranges(image.height)
    for tile_number, (top, bottom) in enumerate(ranges, 1):
        tile_free = available_ram_mb()
        if tile_free < CRITICAL_FREE_RAM_MB:
            gc.collect()
            time.sleep(5)
            if available_ram_mb() < CRITICAL_FREE_RAM_MB:
                raise RuntimeError("Available RAM dropped below critical resource guard during OCR")
        with image.crop((0, top, image.width, bottom)) as tile:
            tile_array = np.asarray(tile)
            predictions = list(ocr.predict(tile_array))
        for prediction in predictions:
            texts = list(prediction.get("rec_texts", []))
            scores = list(prediction.get("rec_scores", []))
            polygons = list(prediction.get("rec_polys", []))
            # The textline angle head reports 0/180 for each recognised line.
            # A caption set at a right angle is deskewed internally by
            # np.rot90 first (see crop_image_regions.get_rotate_crop_image),
            # so this flag is exactly what tells us whether the printed
            # letters run clockwise or counter-clockwise of upright -- and
            # therefore which way the translation has to be rotated back.
            orientation_angles = list(prediction.get("textline_orientation_angles") or [])
            for index, (text, score, polygon) in enumerate(zip(texts, scores, polygons)):
                global_polygon = [[int(x), int(y) + top] for x, y in np.asarray(polygon).tolist()]
                confidence = float(score)
                region = {
                    "id": str(uuid.uuid4()),
                    "text": str(text),
                    "confidence": confidence,
                    "bbox": bbox_from_polygon(global_polygon),
                    "polygon": global_polygon,
                    "orientation_angle": int(orientation_angles[index]) if index < len(orientation_angles) else 0,
                    "type": "dialogue",
                    "status": "OK" if confidence >= CONFIDENCE_THRESHOLD else "LOW_OCR_CONFIDENCE",
                    "tile": tile_number,
                }
                if region["text"].strip() and not is_duplicate(region, regions):
                    regions.append(region)
        del tile_array, predictions
        gc.collect()
        print(f"tile {tile_number}/{len(ranges)} complete", flush=True)

    # A tile that begins inside a caption hands the recogniser a degenerate
    # crop and it answers with garbage at 0.16-0.53 confidence, even though
    # the very same pixels score 0.92-1.00 from a crop whose edge lands on the
    # lettering ("JUST AS I EXPECTED," became 'ISAISN' + 'EAAEEE', and
    # "HEY, DON'T KEEP US WAITING" became 'ANLIAE SE E'). Both were then
    # dropped by the grouping filters, so nothing removed them either and the
    # raw English survived onto the finished page.
    #
    # Every line below the grouping floor is re-read from a crop of its own
    # box at three paddings. Readings are kept only when their centre falls
    # inside this region's box -- a wider crop spans stacked lines and the
    # detector otherwise answers with the neighbour, duplicating the line
    # below -- and the survivors of one crop are joined in reading order so a
    # line the tile shattered into words comes back whole. The crop with the
    # most coverage wins, which is what recovers "HEY, DON'T KEEP US WAITING"
    # from three separate fragments.
    for region in regions:
        if float(region["confidence"]) >= RETRY_CONFIDENCE_BELOW:
            continue
        box = region["bbox"]
        region_letters = len(re.sub(r"[^a-z]", "", region["text"], flags=re.I))
        best_text, best_score, best_letters = region["text"], float(region["confidence"]), region_letters
        for pad in (0, 30, 60):
            x1 = max(0, int(box["x"]) - pad)
            y1 = max(0, int(box["y"]) - pad)
            x2 = min(image.width, int(box["x"]) + int(box["width"]) + pad)
            y2 = min(image.height, int(box["y"]) + int(box["height"]) + pad)
            if x2 <= x1 or y2 <= y1:
                continue
            try:
                retry_predictions = list(ocr.predict(np.asarray(image.crop((x1, y1, x2, y2)))))
            except Exception:
                continue
            collected = []
            for prediction in retry_predictions:
                polys = list(prediction.get("rec_polys", []))
                for index, (text, score) in enumerate(zip(prediction.get("rec_texts", []), prediction.get("rec_scores", []))):
                    text, score = str(text).strip(), float(score)
                    if not text or index >= len(polys):
                        continue
                    poly = np.asarray(polys[index])
                    centre_x = x1 + float(poly[:, 0].mean())
                    centre_y = y1 + float(poly[:, 1].mean())
                    if not (int(box["x"]) <= centre_x <= int(box["x"]) + int(box["width"])
                            and int(box["y"]) <= centre_y <= int(box["y"]) + int(box["height"])):
                        continue
                    collected.append((centre_y, centre_x, score, text))
                break
            if not collected:
                continue
            # Lines are stacked at most half a region height apart, so bucket
            # by that before sorting: same line reads left to right, stacked
            # lines read top to bottom.
            band = max(float(box["height"]) * 0.5, 1.0)
            collected.sort(key=lambda row: (int((row[0] - int(box["y"])) / band), row[1]))
            joined = " ".join(row[3] for row in collected)
            joined_letters = len(re.sub(r"[^a-z]", "", joined, flags=re.I))
            mean_score = sum(row[2] for row in collected) / len(collected)
            if retry_better(joined_letters, mean_score, region_letters, float(region["confidence"])):
                if mean_score > best_score:
                    best_text, best_score, best_letters = joined, mean_score, joined_letters
        if best_score > float(region["confidence"]):
            region["text"] = best_text
            region["confidence"] = best_score
            region["status"] = "OK" if best_score >= CONFIDENCE_THRESHOLD else "LOW_OCR_CONFIDENCE"
            print(f"retry: {best_score:.2f} {best_text!r}", flush=True)

    regions.sort(key=lambda region: (region["bbox"]["y"], region["bbox"]["x"]))
    debug = image.copy()
    draw = ImageDraw.Draw(debug)
    for index, region in enumerate(regions, 1):
        box = region["bbox"]
        color = "lime" if region["status"] == "OK" else "orange"
        draw.rectangle(
            (box["x"], box["y"], box["x"] + box["width"], box["y"] + box["height"]),
            outline=color,
            width=2,
        )
        draw.text((box["x"], max(0, box["y"] - 12)), str(index), fill=color)
    debug_path = args.output_dir / "debug-bboxes.jpg"
    debug.save(debug_path, format="JPEG", quality=92)
    debug.close()

    elapsed = time.perf_counter() - started
    sampler.stop()
    report = {
        "input": str(args.input),
        "format": original_format,
        "width": original_size[0],
        "height": original_size[1],
        "tiles": len(ranges),
        "regions": regions,
        "region_count": len(regions),
        "low_confidence_count": sum(r["status"] != "OK" for r in regions),
        "seconds": round(elapsed, 2),
        "peak_rss_mb": round(sampler.peak_rss_mb, 1),
        "available_ram_before_mb": free_before,
        "minimum_available_ram_mb": sampler.min_available_mb,
        "models": {
            "detection": "PP-OCRv5_mobile_det",
            "recognition": "PP-OCRv5_server_rec",
            "textline_orientation": "PP-LCNet_x1_0_textline_orientation",
        },
    }
    metadata_path = args.output_dir / "ocr-result.json"
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert image.size == original_size
    image.close()
    print(json.dumps({key: value for key, value in report.items() if key != "regions"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
