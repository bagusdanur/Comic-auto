#!/usr/bin/env python3
"""Build a compact, read-only baseline manifest from completed page metadata.

The manifest references existing images instead of copying them, so visual
regression baselines do not consume another multi-gigabyte set of PNG files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_NAMES = ("cleaned.png", "typeset.png", "mask-debug.png", "debug-bboxes.jpg")
REPORT_NAMES = (
    "pipeline-status.json", "ocr-result.json", "translations.json",
    "removal-report.json", "typeset-report.json", "surface-map.json",
    "quality-report.json", "hybrid-report.json", "branding-report.json",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def file_record(path: Path, root: Path, include_hashes: bool) -> dict:
    record = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
    }
    if include_hashes:
        record["sha256"] = digest(path)
    return record


def build_manifest(job_root: Path, include_hashes: bool = False) -> dict:
    root = job_root.resolve()
    metadata = root / "metadata"
    if not metadata.is_dir():
        raise ValueError(f"Job metadata directory not found: {metadata}")
    pages = []
    for page in sorted(metadata.glob("page-*")):
        if not page.is_dir():
            continue
        files = []
        for name in (*IMAGE_NAMES, *REPORT_NAMES):
            candidate = page / name
            if candidate.is_file():
                files.append(file_record(candidate, root, include_hashes))
        status_path = page / "pipeline-status.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        pages.append({
            "page": page.name,
            "status": status.get("status"),
            "pipeline_engine": status.get("pipeline_engine", "legacy-unversioned"),
            "review_ids": status.get("review_ids", []),
            "durations": status.get("durations", {}),
            "files": files,
        })
    return {
        "version": 1,
        "job_root": str(root),
        "page_count": len(pages),
        "include_hashes": include_hashes,
        "pages": pages,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hashes", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest(args.job_root, args.hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "pages": manifest["page_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
