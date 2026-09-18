"""Use a complete text footprint to remove dark outlines as well as fill."""
import json,sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from xianscan_client import XianScanClient
from webp_archive import archive_webp
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
jid='4a075ced-c16a-4d6f-97ba-464db33ef501';job=c.execute('select * from jobs where id=?',(jid,)).fetchone()
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall()
client=XianScanClient('http://127.0.0.1:8123')
for num,start,end in [(3,5481,5633),(6,6513,6663)]:
 row=rows[num-1];p=Path(row['metadata_path']);work=Path('artifacts/ch5-yellow-promo')/p.name
 raw=np.array(Image.open(json.loads((p/'pipeline-status.json').read_text())['input']).convert('RGB'))
 y1=start-500;y2=end+500;context=raw[y1:y2].copy();mask=np.zeros(context.shape[:2],np.uint8)
 mask[start-y1-12:end-y1+14,55:745]=255
 repaired=client.inpaint_mask(context,mask)
 assert not np.any(np.any(repaired!=context,axis=2)&(mask==0))
 targets=[Path(row['output_path'])]+[p/name for name in ('typeset.png','cleaned.png') if (p/name).exists()]
 for i,target in enumerate(targets):
  a=np.array(Image.open(work/('before-output.png' if i==0 else 'before-'+target.name)).convert('RGB'))
  a[y1:y2][mask>0]=repaired[mask>0];Image.fromarray(a).save(target)
 Image.open(targets[0]).crop((0,start-100,800,end+100)).save(work/'final-preview.png')
 print('REFINED',num,flush=True)
print('ARCHIVED',archive_webp([Path(r['output_path']) for r in rows],Path(job['output_path'])),flush=True)
