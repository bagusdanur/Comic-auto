import json
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
db.row_factory = sqlite3.Row
row = db.execute(
    "SELECT id,status,stage,progress,review_count,error,updated_at FROM jobs WHERE id=?",
    (sys.argv[2],),
).fetchone()
print(json.dumps(dict(row) if row else None))
