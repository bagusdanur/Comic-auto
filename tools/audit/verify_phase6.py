import json
import sqlite3
from pathlib import Path

from PIL import Image

root = Path(__file__).resolve().parents[2]
db = sqlite3.connect(root / "data/phase6-test.sqlite3")
db.row_factory = sqlite3.Row
statuses = {row["status"]: row["count"] for row in db.execute(
    "select status, count(*) as count from jobs group by status"
)}
latest = dict(db.execute("select * from jobs order by created_at desc limit 1").fetchone())
with Image.open(root / "output/phase6-queue-final-correct.png") as image:
    output = {"format": image.format, "dimensions": list(image.size)}
print(json.dumps({
    "job_statuses": statuses,
    "processing_jobs": db.execute("select count(*) from jobs where status='PROCESSING'").fetchone()[0],
    "images": db.execute("select count(*) from images").fetchone()[0],
    "regions": db.execute("select count(*) from regions").fetchone()[0],
    "latest_job": {key: latest[key] for key in ("id", "status", "stage", "review_count", "output_path")},
    "requested_output": output,
}, indent=2))

