#!/usr/bin/env python3
"""Remove stored artifacts except the newest N jobs while retaining DB history."""
import argparse
import shutil
import sqlite3
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
parser.add_argument("--jobs-root", type=Path, default=Path("data/jobs"))
parser.add_argument("--keep", type=int, default=5)
parser.add_argument("--execute", action="store_true")
args = parser.parse_args()

if args.keep < 1:
    raise SystemExit("--keep must be at least 1")

root = args.jobs_root.resolve()
database = sqlite3.connect(args.db)
rows = database.execute(
    "SELECT id,status FROM jobs ORDER BY updated_at DESC LIMIT ?", (args.keep,)
).fetchall()
keep_ids = {row[0] for row in rows}
# Never prune a job that is currently capable of writing files.
active_ids = {
    row[0]
    for row in database.execute(
        "SELECT id FROM jobs WHERE status IN ('QUEUED','PROCESSING','REVIEW_REQUIRED')"
    )
}
keep_ids.update(active_ids)

targets = []
for candidate in root.rglob("*"):
    if not candidate.is_dir() or candidate.name in keep_ids:
        continue
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        continue
    # Job artifact directories are named with the UUID stored in jobs.id.
    found = database.execute("SELECT 1 FROM jobs WHERE id=?", (candidate.name,)).fetchone()
    if found:
        targets.append(resolved)

bytes_total = sum(
    item.stat().st_size
    for target in targets
    for item in target.rglob("*")
    if item.is_file()
)
print("KEEP", *sorted(keep_ids), sep="\n")
print(f"TARGET_DIRS={len(targets)} BYTES={bytes_total}")
for target in sorted(targets):
    print("DELETE", target)

if args.execute:
    for target in targets:
        shutil.rmtree(target)
    # Remove only empty hierarchy directories beneath the verified root.
    for directory in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    print("DELETED", len(targets))
