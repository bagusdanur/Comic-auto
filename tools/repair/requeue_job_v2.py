#!/usr/bin/env python3
"""Requeue one existing library job with the quality V2 pipeline enabled."""
import argparse
import json
import sqlite3
import time
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("job_root", type=Path)
parser.add_argument("--apply", action="store_true")
parser.add_argument("--wait", action="store_true")
args = parser.parse_args()

database = sqlite3.connect(args.database, timeout=30)
database.row_factory = sqlite3.Row
needle = f"%{args.job_root.as_posix()}%"
rows = database.execute(
    "SELECT * FROM jobs WHERE input_path LIKE ? ORDER BY created_at DESC", (needle,)
).fetchall()
if len(rows) != 1:
    raise SystemExit(f"Expected exactly one matching job, found {len(rows)}")
row = dict(rows[0])
options = json.loads(row["options_json"])
print(json.dumps({
    "id": row["id"], "status": row["status"], "stage": row["stage"],
    "review_count": row["review_count"], "output_path": row["output_path"],
    "pages": len(options.get("input_files", [])), "quality_v2": options.get("quality_v2"),
}, indent=2))
if args.wait:
    while row["status"] in {"QUEUED", "WAITING_RESOURCE", "PROCESSING"}:
        time.sleep(30)
        row = dict(database.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone())
        print(json.dumps({
            "status": row["status"], "stage": row["stage"],
            "progress": row["progress"], "review_count": row["review_count"],
        }), flush=True)
    raise SystemExit(0)
if not args.apply:
    raise SystemExit(0)
if database.execute("SELECT 1 FROM jobs WHERE status='PROCESSING' AND id<>?", (row["id"],)).fetchone():
    raise SystemExit("Another translation job is processing; refusing to jump the queue")
options["quality_v2"] = "on"
now = time.time()
database.execute(
    "UPDATE jobs SET options_json=?,status='QUEUED',stage='QUEUED_V2_REPROCESS',"
    "progress=0,review_count=0,error=NULL,updated_at=?,started_at=NULL,completed_at=NULL,heartbeat_at=NULL "
    "WHERE id=? AND status<>'PROCESSING'",
    (json.dumps(options), now, row["id"]),
)
database.commit()
print("REQUEUED_V2", row["id"])

