#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from webp_archive import archive_webp

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", type=Path)
parser.add_argument("archive", type=Path)
args = parser.parse_args()
images = sorted(p for p in args.output_dir.iterdir() if p.suffix.casefold() == ".png")
members = archive_webp(images, args.archive)
print(f"rebuilt {args.archive} from {len(images)} pages into {members} WebP files")

