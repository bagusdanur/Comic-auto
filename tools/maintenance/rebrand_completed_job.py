#!/usr/bin/env python3
"""Rebuild branding and archive for an already processed batch without OCR/translation."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from brand_postprocess import apply_branding
from webp_archive import archive_webp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--job-id")
    parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    job_dir = args.job_dir.resolve()
    outputs = sorted(path for path in (job_dir / "output").glob("*.png"))
    pages = sorted((job_dir / "metadata").glob("page-*"))
    if not outputs or len(outputs) != len(pages):
        raise SystemExit(f"output/page mismatch: {len(outputs)} != {len(pages)}")

    for index, (output, page) in enumerate(zip(outputs, pages)):
        source = page / "typeset.png"
        if not source.exists():
            raise SystemExit(f"missing typeset source: {source}")
        shutil.copy2(source, output)
        report = apply_branding(
            output,
            page / "ocr-result.json",
            root / "assets" / "branding" / "watermark.png",
            root / "assets" / "branding" / "watermark-text.png",
            first_page=index == 0,
            banner_path=root / "assets" / "branding" / "banner.png" if index == len(outputs) - 1 else None,
        )
        (page / "branding-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(output.name, len(report.get("replacements", [])), report.get("detected_credit_texts", []))

    archive = job_dir / "output" / "chapter-translated.zip"
    if args.job_id:
        database = sqlite3.connect(args.db)
        row = database.execute("SELECT output_path FROM jobs WHERE id=?", (args.job_id,)).fetchone()
        database.close()
        if not row:
            raise SystemExit(f"unknown job: {args.job_id}")
        archive = Path(row[0])
    archive_webp(outputs, archive)
    # Review thumbnails are generated lazily from the final output. Remove any
    # stale thumbnails after rebuilding branding so Review cannot show the
    # pre-brand image while the ZIP already contains the corrected page.
    previews = job_dir / "output" / "previews"
    if previews.exists():
        for cached in previews.glob("*.jpg"):
            cached.unlink()
    print(f"archive={archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

