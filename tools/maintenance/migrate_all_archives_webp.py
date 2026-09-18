#!/usr/bin/env python3
"""Convert every existing batch download archive to WebP from review masters."""
import argparse
import json
import sqlite3
import zipfile
from pathlib import Path

from webp_archive import archive_webp

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("--apply", action="store_true")
args = parser.parse_args()

database = sqlite3.connect(args.database, timeout=30)
database.row_factory = sqlite3.Row
rows = database.execute(
    "SELECT id,status,output_path,options_json FROM jobs "
    "WHERE status IN ('COMPLETED','REVIEW') ORDER BY created_at"
).fetchall()
summary = {"eligible": 0, "converted": 0, "already_webp": 0, "skipped": [],
           "before_bytes": 0, "after_bytes": 0}
for row in rows:
    options = json.loads(row["options_json"] or "{}")
    archive_path = Path(row["output_path"])
    if not options.get("batch") or archive_path.suffix.casefold() != ".zip":
        continue
    summary["eligible"] += 1
    output_dir = archive_path.parent
    masters = sorted(output_dir.glob("*.png"), key=lambda item: item.name.casefold())
    if not masters:
        summary["skipped"].append({"id": row["id"], "reason": "NO_PNG_MASTERS"})
        continue
    before = archive_path.stat().st_size if archive_path.exists() else 0
    all_webp = False
    if archive_path.exists():
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                all_webp = bool(names) and all(name.casefold().endswith(".webp") for name in names)
        except zipfile.BadZipFile:
            pass
    if all_webp:
        summary["already_webp"] += 1
        summary["before_bytes"] += before
        summary["after_bytes"] += before
        continue
    summary["before_bytes"] += before
    if args.apply:
        archive_webp(masters, archive_path)
        after = archive_path.stat().st_size
        summary["converted"] += 1
        summary["after_bytes"] += after
    else:
        summary["after_bytes"] += before
print(json.dumps(summary, indent=2))

