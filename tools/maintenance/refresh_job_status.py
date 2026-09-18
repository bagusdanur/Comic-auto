#!/usr/bin/env python3
import argparse
import json
import sqlite3
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("job_id")
parser.add_argument("metadata", type=Path)
args = parser.parse_args()

review_count = 0
for status_path in sorted(args.metadata.glob("page-*/pipeline-status.json")):
    status = json.loads(status_path.read_text(encoding="utf-8"))
    review_count += len(status.get("review_ids", []))
status = "REVIEW" if review_count else "COMPLETED"
database = sqlite3.connect(args.database, timeout=30)
database.execute(
    "UPDATE jobs SET status=?,stage='DONE',progress=1,review_count=?,updated_at=? WHERE id=?",
    (status, review_count, time.time(), args.job_id),
)
database.commit()
print(json.dumps({"status": status, "review_count": review_count}))

