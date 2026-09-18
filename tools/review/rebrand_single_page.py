#!/usr/bin/env python3
"""Replace one published page from its typeset artifact and reapply branding."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from brand_postprocess import apply_branding
from webp_archive import archive_webp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("page_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = ROOT
    shutil.copy2(args.page_dir / "typeset.png", args.output)
    report = apply_branding(
        args.output,
        args.page_dir / "ocr-result.json",
        root / "assets/branding/watermark.png",
        root / "assets/branding/watermark-text.png",
        first_page=False,
        banner_path=None,
    )
    (args.page_dir / "branding-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.archive:
        images = sorted(args.output.parent.glob("*.png"))
        archive_webp(images, args.archive)


if __name__ == "__main__":
    main()
