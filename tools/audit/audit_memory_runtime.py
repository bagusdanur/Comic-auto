#!/usr/bin/env python3
"""Read-only summary of translation-memory state and recent trainer jobs."""

import json
import sqlite3
import sys
from pathlib import Path

db = sqlite3.connect("data/translator.sqlite3")
db.row_factory = sqlite3.Row
jobs = [dict(row) for row in db.execute(
    "SELECT id,comic_id,chapter_number,status,stage,progress,error,created_at,updated_at "
    "FROM memory_jobs ORDER BY created_at DESC LIMIT 12"
)]
comics = {row["id"]: dict(row) for row in db.execute("SELECT id,title,slug FROM comics")}
references = [dict(row) for row in db.execute(
    "SELECT comic_id,chapter_number,created_at FROM memory_references ORDER BY created_at DESC LIMIT 12"
)]
memory_root = Path("data/memory")
files = []
if memory_root.exists():
    files = [{"path": str(path), "bytes": path.stat().st_size} for path in sorted(memory_root.rglob("*.json"))]
print(json.dumps({
    "selected_comic": comics.get(sys.argv[1]) if len(sys.argv) > 1 else None,
    "jobs": [{**job, "comic": comics.get(job["comic_id"])} for job in jobs],
    "references": [{**ref, "comic": comics.get(ref["comic_id"])} for ref in references],
    "memory_files": files,
}, ensure_ascii=False, indent=2))
