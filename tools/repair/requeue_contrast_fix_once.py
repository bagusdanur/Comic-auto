import sqlite3
import time

JOB_ID = "d9b1148d-be17-44cb-9d6d-f36adc903b53"
connection = sqlite3.connect("data/translator.sqlite3")
with connection:
    cursor = connection.execute(
        "UPDATE jobs SET status='QUEUED', stage='CONTRAST_FIX_RETRY', progress=0, "
        "error=NULL, started_at=NULL, completed_at=NULL, heartbeat_at=NULL, updated_at=? "
        "WHERE id=? AND status='COMPLETED'",
        (time.time(), JOB_ID),
    )
print(f"requeued={cursor.rowcount}")
