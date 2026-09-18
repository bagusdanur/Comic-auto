#!/usr/bin/env python3
"""Safely requeue one completed translator job for a full current-engine run."""
import argparse
import json
import time
from pathlib import Path

from queue_db import connect

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("job_id")
parser.add_argument("--pipeline-engine", choices=("v3", "v4"))
parser.add_argument("--layout-engine", choices=("v3", "v4"))
parser.add_argument("--branding-enabled", action=argparse.BooleanOptionalAction, default=None)
args = parser.parse_args()

database = connect(args.database)
job_id = args.job_id
row = database.execute("SELECT status,options_json FROM jobs WHERE id=?", (job_id,)).fetchone()
if not row:
    raise SystemExit("job not found")
if row["status"] == "PROCESSING":
    raise SystemExit("job is already processing")
other = database.execute(
    "SELECT id FROM jobs WHERE status='PROCESSING' AND id<>?", (job_id,)
).fetchone()
if other:
    raise SystemExit(f"another job is processing: {other['id']}")
now = time.time()
options = json.loads(row["options_json"] or "{}")
if args.pipeline_engine:
    options["pipeline_engine"] = args.pipeline_engine
if args.layout_engine:
    options["layout_engine"] = args.layout_engine
if args.branding_enabled is not None:
    options["branding_enabled"] = args.branding_enabled
database.execute(
    "UPDATE jobs SET options_json=?,status='QUEUED',stage='FULL_REPROCESS_CURRENT_ENGINE',progress=0,"
    "review_count=0,error=NULL,updated_at=?,started_at=NULL,completed_at=NULL,heartbeat_at=NULL "
    "WHERE id=?",
    (json.dumps(options), now, job_id),
)
database.commit()
print(json.dumps({"requeued": job_id, "options": options}, indent=2))

