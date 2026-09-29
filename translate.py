#!/usr/bin/env python3
"""Phase 5 full CLI: OCR, translate, safely clean, typeset, and export."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from pipeline_engine import KNOWN_ENGINES, require_implemented, requested_engine

# Formats the pipeline can decode. AVIF is decode-only: Mist and other CDNs
# serve AVIF pages and Pillow 12 decodes them natively, but results are always
# written back as PNG/JPEG/WebP, so AVIF is accepted here but never produced.
SUPPORTED_INPUTS = {".png", ".jpg", ".jpeg", ".webp", ".avif"}
SUPPORTED_OUTPUTS = {".png", ".jpg", ".jpeg", ".webp"}


def is_promo_only_page(ocr: dict) -> bool:
    """Recognize a source-credit intro followed by artwork with no dialogue."""
    regions = ocr.get("regions", [])
    height = max(1, int(ocr.get("height", 0)))
    if not regions:
        return False
    substantive = [
        item for item in regions
        if len(re.sub(r"\W+", "", str(item.get("text", "")))) >= 4
        and float(item.get("confidence", 1.0)) >= 0.70
    ]
    if not substantive:
        return False
    # Ignore isolated one-character SFX/noise detected much later in artwork.
    # Those are intentionally filtered from translation and must not turn a
    # source-credit-only intro into a manual review.
    lowest = max(int(item["bbox"]["y"]) + int(item["bbox"]["height"]) for item in substantive)
    combined = " ".join(str(item.get("text", "")) for item in regions)
    has_credit = bool(re.search(
        r"(?:SCAN(?:S|LATION)?|THUNDER|READ\s+(?:ONLY\s+)?AT|WEBSITE|\.COM\b|SUPPORT\s+US)",
        combined,
        re.I,
    ))
    return has_credit and lowest <= height * 0.25


def run_stage(name: str, command: list[str], cwd: Path, status: dict, status_path: Path) -> None:
    status["stage"] = name
    status["status"] = "PROCESSING"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"[{name}] starting", flush=True)
    started = time.perf_counter()
    # Stream child output as before, but retain a bounded tail so the actual
    # provider/service failure reaches pipeline-status.json. A bare
    # CalledProcessError hid HTTP 429 behind "exit status 1", preventing the
    # queue's transient-error retry policy from recognizing it.
    process = subprocess.Popen(
        command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail = deque(maxlen=80)
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line.rstrip())
    returncode = process.wait()
    if returncode:
        detail = "\n".join(tail).strip()
        raise RuntimeError(
            f"{name} command failed with exit status {returncode}"
            + (f":\n{detail}" if detail else "")
        )
    status["durations"][name] = round(time.perf_counter() - started, 2)
    print(f"[{name}] complete in {status['durations'][name]}s", flush=True)


def export_image(source: Path, destination: Path) -> None:
    suffix = destination.suffix.casefold()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".png":
        shutil.copyfile(source, destination)
        return
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        if suffix in {".jpg", ".jpeg"}:
            rgb.save(destination, format="JPEG", quality=95, subsampling=0, optimize=True)
        elif suffix == ".webp":
            rgb.save(destination, format="WEBP", quality=95, method=6)
        else:
            raise ValueError(f"Unsupported output format: {suffix}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provider", default="9router", choices=("9router", "deepseek", "gemini", "openai-compatible"))
    parser.add_argument("--env-file", type=Path, default=Path(".env.9router"))
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--min-y", type=int, default=0)
    parser.add_argument("--font", default="assets/fonts/wildwords/CC Wild Words Roman.ttf")
    parser.add_argument("--min-font-size", type=int, default=22)
    parser.add_argument("--min-scale-ratio", type=float, default=0.52)
    parser.add_argument(
        "--layout-engine", choices=("legacy", "balanced", "v3", "v4", "v5"),
        default=os.getenv("TYPESET_LAYOUT_ENGINE", "legacy").casefold(),
    )
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument(
        "--pipeline-engine", choices=KNOWN_ENGINES,
        default=requested_engine(),
        help="Versioned implementation gate; legacy is the current production pipeline",
    )
    parser.add_argument(
        "--quality-v2", choices=("off", "shadow", "on"),
        default=os.getenv("QUALITY_PIPELINE_V2", "shadow").casefold(),
    )
    parser.add_argument(
        "--hybrid-mode", choices=("off", "shadow", "on", "xianscan", "xianscan-full", "xianscan-mask", "xianscan-v3"),
        default=os.getenv("HYBRID_PIPELINE", "off").casefold(),
    )
    parser.add_argument(
        "--lama-model", type=Path,
        default=Path(os.getenv("LAMA_MODEL_PATH", "models/lama.onnx")),
    )
    args = parser.parse_args()
    pipeline_engine = require_implemented(args.pipeline_engine)
    if pipeline_engine == "v5":
        args.layout_engine = "v5"
        args.quality_v2 = "on"
        args.hybrid_mode = "xianscan-v3"

    root = Path(__file__).resolve().parent
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file() or input_path.suffix.casefold() not in SUPPORTED_INPUTS:
        raise SystemExit("Input must be an existing PNG, JPEG, WebP, or AVIF image")
    # A truncated or empty upload reaches PIL as an opaque UnidentifiedImageError
    # deep inside the OCR stage, making the queue record a confusing OCR failure.
    # Reject it here with a stable reason the worker can surface directly.
    if input_path.stat().st_size == 0:
        raise SystemExit("INPUT_FILE_EMPTY: input image has zero bytes")
    if input_path == output_path:
        raise SystemExit("Refusing to overwrite the original image")
    if output_path.suffix.casefold() not in SUPPORTED_OUTPUTS:
        raise SystemExit("Output extension must be PNG, JPEG, or WebP")

    job_dir = (args.job_dir or root / "output" / f"{input_path.stem}-job").resolve()
    job_dir.mkdir(parents=True, exist_ok=True)
    status_path = job_dir / "pipeline-status.json"
    status = {
        "status": "PROCESSING", "stage": "STARTING", "input": str(input_path),
        "output": str(output_path), "job_dir": str(job_dir), "durations": {},
        "pipeline_engine": pipeline_engine,
    }
    started = time.perf_counter()
    python = sys.executable
    ocr_json = job_dir / "ocr-result.json"
    translations_json = job_dir / "translations.json"
    cleaned = job_dir / "cleaned.png"
    removal_report = job_dir / "removal-report.json"
    typeset = job_dir / "typeset.png"
    typeset_report = job_dir / "typeset-report.json"
    surface_map = job_dir / "surface-map.json"
    quality_report = job_dir / "quality-report.json"
    hybrid_report = job_dir / "hybrid-report.json"

    try:
        run_stage("OCR", [python, str(root / "translate_test.py"), str(input_path), "--output-dir", str(job_dir)], root, status, status_path)
        translation_command = [
            python, str(root / "translate_text_test.py"), str(ocr_json),
            "--provider", args.provider, "--min-y", str(args.min_y),
            "--output", str(translations_json),
        ]
        env_file = args.env_file if args.env_file.is_absolute() else root / args.env_file
        if env_file.exists():
            translation_command += ["--env-file", str(env_file)]
        if args.glossary:
            translation_command += ["--glossary", str(args.glossary.resolve())]
        if args.memory:
            translation_command += ["--memory", str(args.memory.resolve())]
        run_stage("TRANSLATION", translation_command, root, status, status_path)

        translations = json.loads(translations_json.read_text(encoding="utf-8"))
        if not translations.get("groups"):
            # A batch page can legitimately be artwork-only, or OCR may return
            # nothing except low-confidence decoration. This is not a pipeline
            # failure: preserve the page pixel-for-pixel and let the remaining
            # chapter continue. With zero safe groups there is nothing the
            # operator can correct in the translation review UI; SFX/art-only
            # pages must not create permanent false reviews.
            shutil.copyfile(input_path, cleaned)
            shutil.copyfile(input_path, typeset)
            with Image.open(input_path) as source:
                width, height = source.size
                Image.new("L", source.size, 0).save(job_dir / "mask-debug.png", format="PNG")
            empty_report = {
                "input_dimensions": [width, height], "output_dimensions": [width, height],
                "seconds": 0.0, "changed_pixels": 0, "results": [],
            }
            removal_report.write_text(json.dumps({
                **empty_report, "allowed_mask_pixels": 0, "outside_mask_changed_pixels": 0,
                "reason": "NO_SAFE_DIALOGUE_PASSTHROUGH",
            }, indent=2), encoding="utf-8")
            typeset_report.write_text(json.dumps({
                **empty_report, "font": args.font, "outside_typeset_box_changed_pixels": 0,
                "reason": "NO_SAFE_DIALOGUE_PASSTHROUGH",
            }, indent=2), encoding="utf-8")
            export_image(typeset, output_path)
            ocr = json.loads(ocr_json.read_text(encoding="utf-8"))
            promo_only = is_promo_only_page(ocr)
            status.update({
                "status": "COMPLETED", "stage": "DONE",
                "seconds": round(time.perf_counter() - started, 2),
                "dimensions": [ocr["width"], ocr["height"]],
                "ocr_regions": ocr["region_count"], "translated_groups": 0,
                "review_ids": [],
                "peak_rss_mb": ocr["peak_rss_mb"],
                "quality_v2_mode": args.quality_v2, "quality_v2": None,
                "passthrough": True,
                "pipeline_engine": pipeline_engine,
                "passthrough_reason": "SOURCE_PROMO_ONLY" if promo_only else "NO_SAFE_DIALOGUE_PASSTHROUGH",
            })
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status, indent=2))
            return 0
        if args.quality_v2 != "off" or args.hybrid_mode != "off":
            run_stage("SURFACE_ANALYSIS_V2", [
                python, str(root / "quality_pipeline_v2.py"), "analyze",
                str(input_path), str(ocr_json), str(translations_json),
                "--output", str(surface_map),
            ], root, status, status_path)
        removal_command = [
            python, str(root / "remove_text_test.py"), str(input_path), str(ocr_json),
            str(translations_json), "--output", str(cleaned),
            "--mask", str(job_dir / "mask-debug.png"), "--report", str(removal_report),
            "--hybrid-mode", args.hybrid_mode,
            "--lama-model", str((args.lama_model if args.lama_model.is_absolute() else root / args.lama_model).resolve()),
            "--hybrid-report", str(hybrid_report),
            "--xianscan-url", os.getenv("XIANSCAN_ML_URL", "http://127.0.0.1:8123"),
        ]
        if args.quality_v2 == "on" or args.hybrid_mode != "off":
            removal_command += ["--surface-map", str(surface_map)]
        if pipeline_engine == "v5":
            removal_command += ["--strict-artwork-inpaint"]
        run_stage("REMOVAL", removal_command, root, status, status_path)
        typeset_command = [
            python, str(root / "typeset_test.py"), str(cleaned), str(translations_json),
            "--ocr-json", str(ocr_json), "--reference-image", str(input_path),
            "--removal-report", str(removal_report), "--font", args.font,
            "--output", str(typeset), "--report", str(typeset_report),
            "--min-font-size", str(args.min_font_size),
            "--min-scale-ratio", str(args.min_scale_ratio),
            "--layout-engine", args.layout_engine,
        ]
        # Hybrid cleanup already creates a surface map. Always consume it for
        # placement and hard clipping; shadow mode previously analyzed the
        # bubble and then ignored its boundary during typesetting.
        if surface_map.exists():
            typeset_command += ["--surface-map", str(surface_map), "--hard-clip"]
        run_stage("TYPESETTING", typeset_command, root, status, status_path)

        if args.quality_v2 != "off":
            run_stage("QUALITY_V2", [
                python, str(root / "quality_pipeline_v2.py"), "validate",
                str(input_path), str(cleaned), str(typeset), str(job_dir / "mask-debug.png"),
                str(surface_map), "--report", str(quality_report),
            ], root, status, status_path)

        export_image(typeset, output_path)
        with Image.open(input_path) as source, Image.open(output_path) as result:
            if source.size != result.size:
                output_path.unlink(missing_ok=True)
                raise RuntimeError(f"Dimension safety failed: {source.size} != {result.size}")

        ocr = json.loads(ocr_json.read_text(encoding="utf-8"))
        typesetting = json.loads(typeset_report.read_text(encoding="utf-8"))
        # Typesetting is the final authority: removal failures that are safely
        # preserved as tiny SFX should not hold the whole chapter in REVIEW.
        review_ids = sorted({
            item["id"] for item in typesetting["results"] if item["status"] == "NEEDS_REVIEW"
        })
        quality = json.loads(quality_report.read_text(encoding="utf-8")) if quality_report.exists() else None
        if args.quality_v2 == "on" and quality and quality["status"] != "PASS":
            review_ids.extend(f"quality:{reason}" for reason in quality.get("review_reasons", []))
            review_ids = sorted(set(review_ids))
        if pipeline_engine == "v5" and review_ids:
            raise RuntimeError("V5 automatic quality check failed: " + ", ".join(review_ids))
        status.update({
            "status": "REVIEW" if review_ids else "COMPLETED", "stage": "DONE",
            "seconds": round(time.perf_counter() - started, 2),
            "dimensions": [ocr["width"], ocr["height"]],
            "ocr_regions": ocr["region_count"], "translated_groups": len(translations["groups"]),
            "review_ids": review_ids, "peak_rss_mb": ocr["peak_rss_mb"],
            "quality_v2_mode": args.quality_v2,
            "quality_v2": quality,
            "hybrid_mode": args.hybrid_mode,
            "pipeline_engine": pipeline_engine,
            "layout_engine": args.layout_engine,
            "hybrid": json.loads(hybrid_report.read_text(encoding="utf-8")) if hybrid_report.exists() else None,
        })
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 0
    except Exception as error:
        status.update({"status": "FAILED", "error": str(error), "seconds": round(time.perf_counter() - started, 2)})
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(f"Pipeline failed safely: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
