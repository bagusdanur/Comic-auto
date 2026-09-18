#!/usr/bin/env python3
"""Fast re-typeset of an existing job using Typeset Engine V4."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from brand_postprocess import apply_branding
from queue_db import connect
from queue_worker import persist_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    database = connect(root / "data" / "translator.sqlite3")
    job = database.execute("SELECT * FROM jobs WHERE id=?", (args.job_id,)).fetchone()
    if not job:
        raise SystemExit(f"Job {args.job_id} not found in database")

    options = json.loads(job["options_json"])
    job_root = Path(job["input_path"]).parent.parent
    originals = [job_root / "original" / name for name in options.get("input_files", [])]
    if not originals:
        originals = sorted((job_root / "original").glob("*"))

    print(f"Starting V4 re-typeset for job: {args.job_id} ({len(originals)} pages)...", flush=True)
    work = Path(tempfile.mkdtemp(prefix=f"v4-retypeset-{args.job_id[:8]}-"))
    rendered_pages: list[tuple[Path, Path]] = []

    try:
        for index, source in enumerate(originals, 1):
            current = job_root / "metadata" / f"page-{index:03d}"
            page = work / f"page-{index:03d}"
            page.mkdir(parents=True, exist_ok=True)
            cleaned_png = page / "cleaned.png"
            mask_png = page / "mask-debug.png"
            removal_json = page / "removal-report.json"
            output_png = page / f"{source.stem}.png"
            typeset_json = page / "typeset-report.json"

            # Re-clean using updated border-safe logic (erases glyphs, zero white box overflow)
            clean_cmd = [
                sys.executable, str(root / "remove_text_test.py"),
                str(source),
                str(current / "ocr-result.json"),
                str(current / "translations.json"),
                "--output", str(cleaned_png),
                "--mask", str(mask_png),
                "--report", str(removal_json),
                "--surface-map", str(current / "surface-map.json"),
                "--hybrid-mode", "xianscan-v3",
                "--lama-model", str(root / "models" / "lama.onnx"),
            ]
            subprocess.run(clean_cmd, cwd=root, check=True, stdout=subprocess.DEVNULL)

            cmd = [
                sys.executable, str(root / "typeset_test.py"),
                str(cleaned_png),
                str(current / "translations.json"),
                "--ocr-json", str(current / "ocr-result.json"),
                "--reference-image", str(source),
                "--removal-report", str(removal_json),
                "--surface-map", str(current / "surface-map.json"),
                "--hard-clip",
                "--layout-engine", "v4",
                "--output", str(output_png),
                "--report", str(typeset_json),
            ]
            subprocess.run(cmd, cwd=root, check=True, stdout=subprocess.DEVNULL)

            # Apply branding (replace raw logos with Ryukomik logo)
            branding = apply_branding(
                output_png, current / "ocr-result.json",
                root / "assets/branding/watermark.png",
                root / "assets/branding/watermark-text.png",
                first_page=(index == 1),
                banner_path=(root / "assets/branding/banner.png") if index == len(originals) else None,
            )
            (page / "branding-report.json").write_text(json.dumps(branding, indent=2), encoding="utf-8")

            destination = job_root / "output" / f"{source.stem}.png"
            rendered_pages.append((page, destination))
            print(f"  Cleaned, typeset and branded page {index}/{len(originals)}", flush=True)

        # Copy rendered files back to metadata and output
        for index, (page, destination) in enumerate(rendered_pages, 1):
            current = job_root / "metadata" / f"page-{index:03d}"
            for name in ("cleaned.png", "mask-debug.png", "removal-report.json", "typeset-report.json", "branding-report.json"):
                if (page / name).exists():
                    shutil.copy2(page / name, current / name)
            shutil.copy2(page / destination.name, destination)

        # Re-pack chapter-translated.zip
        archive_path = Path(job["output_path"])
        temporary_archive = work / "chapter-translated.zip"
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for _, destination in rendered_pages:
                archive.write(destination, arcname=destination.name)
        temporary_archive.replace(archive_path)

        # Update database entries
        database.execute("DELETE FROM images WHERE job_id=?", (args.job_id,))
        review_count = 0
        for index, (_, destination) in enumerate(rendered_pages, 1):
            metadata = job_root / "metadata" / f"page-{index:03d}"
            report = json.loads((metadata / "typeset-report.json").read_text(encoding="utf-8"))
            page_review = sum(item.get("status") == "NEEDS_REVIEW" for item in report.get("results", []))
            review_count += page_review
            orig_names = options.get("original_names", [])
            filename = orig_names[index - 1] if len(orig_names) == len(originals) else originals[index - 1].name
            persist_results(database, {"id": args.job_id}, metadata, destination, filename, "REVIEW" if page_review else "COMPLETED")

        now = time.time()
        database.execute(
            "UPDATE jobs SET status=?, review_count=?, updated_at=? WHERE id=?",
            ("REVIEW" if review_count else "COMPLETED", review_count, now, args.job_id)
        )
        database.commit()
        shutil.rmtree(work, ignore_errors=True)
        print(json.dumps({
            "status": "SUCCESS",
            "job_id": args.job_id,
            "pages": len(originals),
            "review_count": review_count,
        }, indent=2))
        return 0
    except Exception as e:
        database.rollback()
        shutil.rmtree(work, ignore_errors=True)
        print(f"Error re-typesetting: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
