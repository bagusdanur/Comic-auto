#!/usr/bin/env python3
"""Requeue one explicitly selected failed job for an idempotent resume."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from queue_db import connect


parser = argparse.ArgumentParser()
parser.add_argument("job_id")
parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
parser.add_argument(
    "--from-status", choices=("FAILED", "CANCELLED", "COMPLETED"), default="FAILED",
    help="Explicit current state required before requeueing the selected job.",
)
parser.add_argument(
    "--fresh", action="store_true",
    help="Discard image/region database rows so the selected job rebuilds from page one.",
)
args = parser.parse_args()
database = connect(args.db)
updated = database.execute(
    "UPDATE jobs SET status='QUEUED',stage='MANUAL_FIXED_RETRY',progress=0,"
    "error=NULL,started_at=NULL,completed_at=NULL,heartbeat_at=NULL,updated_at=? "
    "WHERE id=? AND status=?",
    (time.time(), args.job_id, args.from_status),
)
database.commit()
row = database.execute("SELECT id,status,stage FROM jobs WHERE id=?", (args.job_id,)).fetchone()
if updated.rowcount != 1 or row is None:
    raise SystemExit(f"Job was not found in {args.from_status} state")
if args.fresh:
    database.execute("DELETE FROM images WHERE job_id=?", (args.job_id,))
    database.commit()
print(dict(row))
