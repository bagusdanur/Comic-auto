#!/usr/bin/env python3
"""Requeue one failed memory job while preserving its uploaded archives."""

import argparse
import sqlite3
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_id")
parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
args = parser.parse_args()
db = sqlite3.connect(args.db)
row = db.execute("SELECT status,raw_zip_path,translated_zip_path FROM memory_jobs WHERE id=?", (args.job_id,)).fetchone()
if not row:
    raise SystemExit("memory job not found")
if row[0] != "FAILED":
    raise SystemExit(f"job is {row[0]}, expected FAILED")
for archive in row[1:]:
    if not archive or not Path(archive).is_file():
        raise SystemExit(f"stored upload missing: {archive}")
db.execute(
    "UPDATE memory_jobs SET status='QUEUED',stage='QUEUED_RETRY',progress=0,error=NULL,"
    "started_at=NULL,heartbeat_at=NULL,completed_at=NULL,updated_at=? WHERE id=?",
    (time.time(), args.job_id),
)
db.commit()
print(args.job_id, "QUEUED_RETRY")
