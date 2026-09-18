#!/usr/bin/env python3
import argparse
import sqlite3
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("job_id")
args = parser.parse_args()
database = sqlite3.connect(args.database, timeout=30)
rows = database.execute("SELECT id,output_path FROM images WHERE job_id=?", (args.job_id,)).fetchall()
removed = 0
for image_id, output_path in rows:
    preview = Path(output_path).parent / "previews" / f"{image_id}.jpg"
    if preview.exists():
        preview.unlink()
        removed += 1
print(f"invalidated {removed} generated previews for {len(rows)} images")

