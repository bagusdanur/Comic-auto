#!/usr/bin/env python3
"""Re-render an existing completed job with the current cleanup/typeset pipeline."""

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

from brand_postprocess import apply_branding
from queue_db import connect
from queue_worker import persist_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    database = connect(root / "data" / "translator.sqlite3")
    job = database.execute("SELECT * FROM jobs WHERE id=?", (args.job_id,)).fetchone()
    if not job:
        raise SystemExit("Job not found")
    options = json.loads(job["options_json"])
    job_root = Path(job["input_path"]).parent.parent
    originals = [job_root / "original" / name for name in options.get("input_files", [])]
    if not originals:
        originals = sorted((job_root / "original").glob("*"))
    backup = Path(tempfile.mkdtemp(prefix=f"translator-upgrade-{args.job_id[:8]}-"))
    shutil.copytree(job_root / "output", backup / "output")
    work = backup / "work"
    work.mkdir()
    rendered_pages: list[tuple[Path, Path]] = []
    try:
        for index, source in enumerate(originals, 1):
            current = job_root / "metadata" / f"page-{index:03d}"
            page = work / f"page-{index:03d}"
            page.mkdir()
            cleaned, mask = page / "cleaned.png", page / "mask-debug.png"
            removal, typeset = page / "removal-report.json", page / "typeset-report.json"
            output = page / f"{source.stem}.png"
            subprocess.run([
                sys.executable, str(root / "remove_text_test.py"), str(source),
                str(current / "ocr-result.json"), str(current / "translations.json"),
                "--output", str(cleaned), "--mask", str(mask), "--report", str(removal),
            ], cwd=root, check=True, stdout=subprocess.DEVNULL)
            subprocess.run([
                sys.executable, str(root / "typeset_test.py"), str(cleaned),
                str(current / "translations.json"), "--ocr-json", str(current / "ocr-result.json"),
                "--reference-image", str(source), "--removal-report", str(removal),
                "--output", str(output), "--report", str(typeset),
            ], cwd=root, check=True, stdout=subprocess.DEVNULL)
            branding = apply_branding(
                output, current / "ocr-result.json", root / "assets/branding/watermark.png",
                root / "assets/branding/watermark-text.png", first_page=index == 1,
                banner_path=(root / "assets/branding/banner.png") if index == len(originals) else None,
            )
            (page / "branding-report.json").write_text(json.dumps(branding, indent=2), encoding="utf-8")
            destination = job_root / "output" / f"{source.stem}.png"
            rendered_pages.append((page, destination))
            print(f"rendered {index}/{len(originals)}", flush=True)

        for index, (page, destination) in enumerate(rendered_pages, 1):
            current = job_root / "metadata" / f"page-{index:03d}"
            for name in ("cleaned.png", "mask-debug.png", "removal-report.json", "typeset-report.json", "branding-report.json"):
                shutil.copy2(page / name, current / name)
            shutil.copy2(page / destination.name, destination)

        archive_path = Path(job["output_path"])
        temporary_archive = work / "chapter-translated.zip"
        with zipfile.ZipFile(temporary_archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for _, destination in rendered_pages:
                archive.write(destination, arcname=destination.name)
        temporary_archive.replace(archive_path)

        database.execute("DELETE FROM images WHERE job_id=?", (args.job_id,))
        review_count = 0
        for index, (_, destination) in enumerate(rendered_pages, 1):
            metadata = job_root / "metadata" / f"page-{index:03d}"
            report = json.loads((metadata / "typeset-report.json").read_text(encoding="utf-8"))
            page_review = sum(item["status"] == "NEEDS_REVIEW" for item in report["results"])
            review_count += page_review
            filename = options.get("original_names", [])[index - 1] if len(options.get("original_names", [])) == len(originals) else originals[index - 1].name
            persist_results(database, {"id": args.job_id}, metadata, destination, filename, "REVIEW" if page_review else "COMPLETED")
        now = time.time()
        database.execute("UPDATE jobs SET status=?,review_count=?,updated_at=? WHERE id=?", ("REVIEW" if review_count else "COMPLETED", review_count, now, args.job_id))
        database.commit()
        print(json.dumps({"job_id": args.job_id, "pages": len(originals), "review_count": review_count, "backup": str(backup)}))
        return 0
    except Exception:
        database.rollback()
        raise


if __name__ == "__main__":
    raise SystemExit(main())

