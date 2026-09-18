"""Repair the verified stacked promotional header, preserving translated pixels."""
import json
import shutil
import sqlite3
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image
from branding_v52 import card_bounds
from webp_archive import archive_webp

jid='6683eec6-57f9-4283-b83d-9d7911b7233e'
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
job=c.execute('select * from jobs where id=?',(jid,)).fetchone()
assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall()
r=rows[0];out=Path(r['output_path']);page=Path(r['metadata_path'])
if page.is_file():page=page.parent
report_path=page/'branding-report.json'
report=json.loads(report_path.read_text())
assert not report.get('source_header_trimmed_at'), 'Already cropped'
raw=Path(json.loads((page/'pipeline-status.json').read_text())['input'])
with Image.open(raw) as im: bounds=card_bounds(np.array(im.convert('RGB')),json.loads((page/'ocr-result.json').read_text())['regions'])
assert bounds==(1520,14370), bounds
backup=Path('artifacts/ch4-header-before');backup.mkdir(exist_ok=True)
for path in [out,report_path,Path(job['output_path'])]:
 assert not (backup/path.name).exists(), 'Backup already exists'
 shutil.copy2(path,backup/path.name)
with Image.open(out) as im:
 assert im.size==(800,14370)
 cropped=im.crop((0,bounds[0],800,bounds[1]))
 staged=out.with_suffix('.header-fixed.png');cropped.save(staged)
 cropped.crop((0,0,800,1000)).save(backup/'after-top.png')
report['source_header_trimmed_at']=bounds[0]
report['output_dimensions']=list(cropped.size)
report['header_crop_guard']='stacked-warning-v1'
for mark in report.get('replacements',[]):mark['logo'][1]-=bounds[0]
archive=Path(job['output_path']);staged_zip=archive.with_suffix('.header-fixed.zip')
with zipfile.ZipFile(staged_zip,'w',zipfile.ZIP_DEFLATED) as z:
 for row in rows:
  p=Path(row['output_path']);z.write(staged if p==out else p,p.name)
with zipfile.ZipFile(staged_zip) as z:assert len(z.namelist())==9 and z.testzip() is None
staged.replace(out);staged_zip.replace(archive)
archive_webp([Path(row['output_path']) for row in rows], archive)
report_path.write_text(json.dumps(report,indent=2))
c.execute('update images set width=?,height=? where id=?',(*cropped.size,r['id']));c.commit()
print('REPAIRED',r['id'],cropped.size,'ZIP 9 valid; originals backed up')
