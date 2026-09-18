"""Explicit repair entry point. Pause the normal worker before running this."""
import argparse
import json
import time
from pathlib import Path
from queue_db import connect
from queue_worker import process_job

p=argparse.ArgumentParser();p.add_argument('job_id');args=p.parse_args()
root=Path.cwd();db=connect(root/'data/translator.sqlite3')
row=db.execute('SELECT * FROM jobs WHERE id=?',(args.job_id,)).fetchone()
if not row or row['status']!='FAILED':raise SystemExit('Only a failed job may be resumed')
if db.execute("SELECT count(*) FROM jobs WHERE status='PROCESSING'").fetchone()[0]:raise SystemExit('Another job is active')
job=dict(row);job['options']=json.loads(job['options_json'])
backup=root/'artifacts'/'resume-branding'/args.job_id;backup.mkdir(parents=True,exist_ok=True)
if not (backup/'job-before.json').exists():(backup/'job-before.json').write_text(json.dumps(dict(row),indent=2))
now=time.time()
changed=db.execute("UPDATE jobs SET status='PROCESSING',stage='RESUME BRANDING',error=NULL,completed_at=NULL,updated_at=?,heartbeat_at=? WHERE id=? AND status='FAILED'",(now,now,args.job_id)).rowcount
if changed!=1:raise SystemExit('Job changed while acquiring repair')
db.commit()
try:
    process_job(db,job,root,resume=True)
except Exception as exc:
    db.rollback();now=time.time()
    db.execute("UPDATE jobs SET status='FAILED',stage='FAILED',error=?,updated_at=?,completed_at=?,heartbeat_at=NULL WHERE id=? AND status='PROCESSING'",(str(exc)[:2000],now,now,args.job_id));db.commit()
    raise
print('RESUME_RESULT',dict(db.execute('SELECT id,status,progress,review_count,error FROM jobs WHERE id=?',(args.job_id,)).fetchone()),flush=True)
