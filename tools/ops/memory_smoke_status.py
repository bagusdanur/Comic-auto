import json
import sqlite3

database = sqlite3.connect("data/translator.sqlite3")
database.row_factory = sqlite3.Row
row = database.execute(
    "SELECT id,status,stage,progress,error,proposal_json FROM memory_jobs WHERE id=?",
    ("093e5287-5c58-4772-a414-597bb3db73f2",),
).fetchone()
result = dict(row)
if result.get("proposal_json"):
    proposal = json.loads(result.pop("proposal_json"))
    result["proposal"] = {
        "pair_count": proposal.get("pair_count"),
        "terminology_count": len(proposal.get("terminology", [])),
        "summary": proposal.get("summary"),
    }
print(json.dumps(result, ensure_ascii=False))

