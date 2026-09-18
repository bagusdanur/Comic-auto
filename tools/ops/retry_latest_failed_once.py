import sqlite3
import time

connection = sqlite3.connect("data/translator.sqlite3")
row = connection.execute("SELECT id FROM jobs WHERE status='FAILED' ORDER BY created_at DESC LIMIT 1").fetchone()
if not row:
    print("requeued=0")
else:
    with connection:
        result = connection.execute(
            "UPDATE jobs SET status='QUEUED',stage='TRANSLATION_RECOVERY',progress=0,error=NULL,"
            "started_at=NULL,completed_at=NULL,heartbeat_at=NULL,updated_at=? WHERE id=? AND status='FAILED'",
            (time.time(), row[0]),
        )
    print(f"requeued={result.rowcount} job={row[0]}")
