#!/usr/bin/env python3
"""Rebuild a chapter ZIP from its rendered page directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from webp_archive import archive_webp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    pages = sorted(path for path in args.output_dir.glob("*.png") if path.is_file())
    if not pages:
        raise SystemExit("No rendered PNG pages found")
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    archive_webp(pages, args.archive)
    print(f"archive={args.archive} pages={len(pages)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

