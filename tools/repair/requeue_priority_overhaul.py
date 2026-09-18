"""One-time backed-up reprocessing for the user-prioritized Chapter 37."""
import json,shutil,time
from pathlib import Path
from queue_db import connect
jid='adbfe8cc-304c-4fdc-8287-40e6ac878345'
c=connect(Path('data/translator.sqlite3'));job=c.execute('select * from jobs where id=?',(jid,)).fetchone()
assert job['status']=='COMPLETED'
assert not c.execute("select 1 from jobs where status='PROCESSING'").fetchone()
root=Path(job['input_path']).parent.parent
backup=Path('artifacts/priority-overhaul-before')/jid
assert not backup.exists(), 'One backup only; inspect existing state before retrying'
backup.mkdir(parents=True)
(backup/'job.json').write_text(json.dumps(dict(job),indent=2))
images=[dict(r) for r in c.execute('select * from images where job_id=?',(jid,))]
regions=[dict(r) for r in c.execute('select r.* from regions r join images i on i.id=r.image_id where i.job_id=?',(jid,))]
(backup/'images.json').write_text(json.dumps(images,indent=2));(backup/'regions.json').write_text(json.dumps(regions,indent=2))
shutil.copytree(root/'metadata',backup/'metadata');shutil.copytree(root/'output',backup/'output')
options=json.loads(job['options_json']);options['priority_overhaul']='seams-open-bubble-vortex-v1'
assert (root/'prepared-seams-v1/manifest.json').exists()
c.execute("update jobs set status='QUEUED',stage='PRIORITY_REBUILD',progress=0,review_count=0,error=NULL,started_at=NULL,completed_at=NULL,heartbeat_at=NULL,updated_at=?,options_json=? where id=? and status='COMPLETED'",(time.time(),json.dumps(options),jid));c.commit()
print('QUEUED',jid,'BACKUP',backup,flush=True)
