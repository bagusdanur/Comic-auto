import json
import sqlite3
import sys
import time

database_path, job_id = sys.argv[1:3]
last = None
while True:
    db = sqlite3.connect(database_path)
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT status,stage,progress,review_count,error FROM jobs WHERE id=?", (job_id,)
    ).fetchone()
    db.close()
    state = dict(row) if row else {"status": "MISSING"}
    encoded = json.dumps(state)
    if encoded != last:
        print(encoded, flush=True)
        last = encoded
    if state.get("status") not in {"QUEUED", "WAITING_RESOURCE", "PROCESSING"}:
        break
    time.sleep(10)

