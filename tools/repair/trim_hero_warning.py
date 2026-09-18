import json,sqlite3,shutil,zipfile
from pathlib import Path
from PIL import Image
import numpy as np
from branding_v52 import card_bounds
from webp_archive import archive_webp
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
jid='92217e9b-e48e-4b53-8e70-8e93b7ca61c6';job=c.execute('select * from jobs where id=?',(jid,)).fetchone();assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall();r=rows[0];p=Path(r['metadata_path']);out=Path(r['output_path'])
report=json.loads((p/'branding-report.json').read_text());assert not report.get('source_header_trimmed_at')
raw=json.loads((p/'pipeline-status.json').read_text())['input']
with Image.open(raw) as im:top,_=card_bounds(np.array(im.convert('RGB')),json.loads((p/'ocr-result.json').read_text())['regions'])
assert top==1521
work=Path('artifacts/hero-warning-fix');work.mkdir(exist_ok=False)
shutil.copy2(out,work/'before.png');shutil.copy2(job['output_path'],work/'before.zip');shutil.copy2(p/'branding-report.json',work/'before-report.json')
regions=[dict(x) for x in c.execute('select * from regions where image_id=?',(r['id'],))];(work/'before-regions.json').write_text(json.dumps(regions))
with Image.open(out) as im:
 im=im.crop((0,top,im.width,im.height));im.save(work/'after.png');size=im.size
report['source_header_trimmed_at']=top;report['output_dimensions']=list(size)
for mark in report.get('replacements',[]):
 if 'logo' in mark:mark['logo'][1]-=top
for row in regions:
 box=json.loads(row['bbox_json']);meta=json.loads(row['metadata_json'])
 if box['y']+box['height']<=top:c.execute('delete from regions where id=?',(row['id'],));continue
 box['y']-=top
 if 'bbox' in meta:meta['bbox']['y']-=top
 c.execute('update regions set bbox_json=?,metadata_json=? where id=?',(json.dumps(box),json.dumps(meta),row['id']))
shutil.copy2(work/'after.png',out);(p/'branding-report.json').write_text(json.dumps(report,indent=2))
archive_webp([Path(row['output_path']) for row in rows],Path(job['output_path']))
c.execute('update images set width=?,height=?,region_count=(select count(*) from regions where image_id=?) where id=?',(*size,r['id'],r['id']));c.commit()
preview=out.parent/'previews'/f"{r['id']}.jpg"
if preview.exists():shutil.move(str(preview),str(work/'before-preview.jpg'))
with zipfile.ZipFile(job['output_path']) as z:assert z.testzip() is None;print('PUBLISHED',len(z.namelist()),'WebP files; removed header',top)
