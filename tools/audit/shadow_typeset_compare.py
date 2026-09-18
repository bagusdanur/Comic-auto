#!/usr/bin/env python3
"""Run a serial, storage-bounded legacy/balanced typeset comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw


COMPARE_FIELDS = ("status", "reason", "wrapped", "font_size", "line_count", "bbox")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def result_map(report: dict) -> dict[str, dict]:
    return {str(item.get("id")): item for item in report.get("results", [])}


def changed_fields(before: dict, after: dict) -> list[str]:
    return [field for field in COMPARE_FIELDS if before.get(field) != after.get(field)]


def union_box(first: dict, second: dict, width: int, height: int, margin: int = 80):
    boxes = [item for item in (first, second) if item]
    x1 = max(0, min(int(item.get("x", 0)) for item in boxes) - margin)
    y1 = max(0, min(int(item.get("y", 0)) for item in boxes) - margin)
    x2 = min(width, max(int(item.get("x", 0)) + int(item.get("width", 1)) for item in boxes) + margin)
    y2 = min(height, max(int(item.get("y", 0)) + int(item.get("height", 1)) for item in boxes) + margin)
    return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)


def save_pair(legacy: Image.Image, balanced: Image.Image, box, label: str, output: Path) -> None:
    left, right = legacy.crop(box), balanced.crop(box)
    header = 34
    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height) + header), "white")
    canvas.paste(left.convert("RGB"), (0, header))
    canvas.paste(right.convert("RGB"), (left.width, header))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"LEGACY | {label}", fill="black")
    draw.text((left.width + 8, 8), "BALANCED", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="WEBP", quality=88, method=6)


def make_contact_sheet(crops: list[Path], output: Path) -> None:
    if not crops:
        return
    thumbs = []
    for path in crops:
        image = Image.open(path).convert("RGB")
        image.thumbnail((700, 420), Image.Resampling.LANCZOS)
        thumbs.append((path.stem, image.copy()))
    cell_width, cell_height, columns = 720, 470, 2
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (cell_width * columns, cell_height * rows), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(thumbs):
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        draw.text((x + 10, y + 8), name, fill="black")
        sheet.paste(image, (x + 10, y + 32))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=88, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-font-size", type=int, default=22)
    parser.add_argument("--min-scale-ratio", type=float, default=0.52)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[2]
    pages = sorted((args.job_root / "metadata").glob("page-*"))
    if args.limit:
        pages = pages[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    crops: list[Path] = []
    page_summaries = []
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="typeset-shadow-") as temporary:
        temp = Path(temporary)
        for page in pages:
            required = {
                "cleaned": page / "cleaned.png", "translations": page / "translations.json",
                "ocr": page / "ocr-result.json", "reference": load(page / "pipeline-status.json").get("input"),
                "removal": page / "removal-report.json", "surface": page / "surface-map.json",
                "legacy_image": page / "typeset.png", "legacy_report": page / "typeset-report.json",
            }
            missing = [name for name, path in required.items() if not path or not Path(path).is_file()]
            if missing:
                page_summaries.append({"page": page.name, "error": f"missing: {', '.join(missing)}"})
                continue
            balanced_image = temp / f"{page.name}.png"
            balanced_report = temp / f"{page.name}.json"
            command = [
                sys.executable, str(project / "typeset_test.py"), str(required["cleaned"]),
                str(required["translations"]), "--ocr-json", str(required["ocr"]),
                "--reference-image", str(required["reference"]), "--removal-report", str(required["removal"]),
                "--surface-map", str(required["surface"]), "--hard-clip", "--layout-engine", "balanced",
                "--min-font-size", str(args.min_font_size), "--min-scale-ratio", str(args.min_scale_ratio),
                "--output", str(balanced_image), "--report", str(balanced_report),
            ]
            run = subprocess.run(command, cwd=project, capture_output=True, text=True)
            if run.returncode:
                page_summaries.append({"page": page.name, "error": run.stderr[-1200:] or run.stdout[-1200:]})
                continue
            legacy_report, candidate_report = load(required["legacy_report"]), load(balanced_report)
            before, after = result_map(legacy_report), result_map(candidate_report)
            changes = []
            with Image.open(required["legacy_image"]) as legacy, Image.open(balanced_image) as balanced:
                for region_id in sorted(set(before) | set(after)):
                    old, new = before.get(region_id, {}), after.get(region_id, {})
                    fields = changed_fields(old, new)
                    if not fields:
                        continue
                    crop_path = args.output / "crops" / f"{page.name}-{region_id}.webp"
                    box = union_box(old.get("bbox", {}), new.get("bbox", {}), legacy.width, legacy.height)
                    save_pair(legacy, balanced, box, region_id, crop_path)
                    crops.append(crop_path)
                    changes.append({"id": region_id, "fields": fields, "legacy": old, "balanced": new})
            page_summaries.append({
                "page": page.name, "seconds": candidate_report.get("seconds"),
                "outside_changed_pixels": candidate_report.get("outside_typeset_box_changed_pixels"),
                "changed_regions": len(changes), "changes": changes,
            })

    regressions = sum(
        old.get("status") == "TYPESET" and new.get("status") == "NEEDS_REVIEW"
        for page in page_summaries for change in page.get("changes", [])
        for old, new in [(change["legacy"], change["balanced"])]
    )
    improvements = sum(
        old.get("status") == "NEEDS_REVIEW" and new.get("status") == "TYPESET"
        for page in page_summaries for change in page.get("changes", [])
        for old, new in [(change["legacy"], change["balanced"])]
    )
    summary = {
        "version": 1, "candidate": "balanced", "production_unchanged": True,
        "job_root": str(args.job_root.resolve()), "pages_requested": len(pages),
        "pages_completed": sum("error" not in page for page in page_summaries),
        "pages_failed": sum("error" in page for page in page_summaries),
        "changed_regions": sum(page.get("changed_regions", 0) for page in page_summaries),
        "regressions": regressions, "improvements": improvements,
        "outside_changed_pixels": sum(page.get("outside_changed_pixels") or 0 for page in page_summaries),
        "wall_seconds": round(time.perf_counter() - started, 2), "pages": page_summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_contact_sheet(crops, args.output / "contact-sheet.jpg")
    print(json.dumps({key: value for key, value in summary.items() if key != "pages"}, ensure_ascii=False))
    return 1 if summary["pages_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
