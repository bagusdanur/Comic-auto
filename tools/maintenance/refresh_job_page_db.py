#!/usr/bin/env python3
"""Replace one persisted image/region set after a targeted page rerender."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from queue_db import connect
from queue_worker import persist_results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_id")
    parser.add_argument("metadata", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    database = connect(root / "data" / "translator.sqlite3")
    old = database.execute(
        "SELECT id,filename FROM images WHERE job_id=? AND metadata_path=?",
        (args.job_id, str(args.metadata.resolve())),
    ).fetchone()
    if old is None:
        raise SystemExit("persisted page not found")
    with database:
        database.execute("DELETE FROM images WHERE id=?", (old["id"],))
        persist_results(
            database, {"id": args.job_id}, args.metadata.resolve(),
            args.output.resolve(), old["filename"], "COMPLETED",
        )
        review_count = database.execute(
            "SELECT count(*) count FROM regions r JOIN images i ON i.id=r.image_id "
            "WHERE i.job_id=? AND r.status='NEEDS_REVIEW'", (args.job_id,),
        ).fetchone()["count"]
        database.execute(
            "UPDATE jobs SET review_count=?,status=?,updated_at=? WHERE id=?",
            (review_count, "REVIEW" if review_count else "COMPLETED", time.time(), args.job_id),
        )
    print(f"page={old['filename']} review_count={review_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

