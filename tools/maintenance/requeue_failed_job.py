#!/usr/bin/env python3
"""Safely resume one failed job from its completed-page checkpoints."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from queue_db import connect


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
    args = parser.parse_args()

    database = connect(args.db)
    now = time.time()
    updated = database.execute(
        "UPDATE jobs SET status='QUEUED',stage='RETRY_FROM_CHECKPOINT',error=NULL,"
        "updated_at=?,completed_at=NULL,heartbeat_at=NULL WHERE id=? AND status='FAILED'",
        (now, args.job_id),
    )
    database.commit()
    if updated.rowcount != 1:
        raise SystemExit("Job not found or is not FAILED")
    print(f"{args.job_id}: QUEUED (resume completed pages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
