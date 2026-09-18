import json,shutil,sqlite3,zipfile
from pathlib import Path
from PIL import Image
from brand_postprocess import apply_branding
from webp_archive import archive_webp
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
jid='b98aa2d6-9758-45b7-8d0c-ca0ee8365ecc'
job=c.execute('select * from jobs where id=?',(jid,)).fetchone();assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall()
work=Path('artifacts/onimusha-transparent-v1');work.mkdir(exist_ok=False)
staged=[]
for i,r in enumerate(rows):
 p=Path(r['metadata_path']);out=work/Path(r['output_path']).name;shutil.copy2(p/'typeset.png',out)
 report=apply_branding(out,p/'ocr-result.json',Path('assets/branding/watermark.png'),Path('assets/branding/watermark-text.png'),banner_path=Path('assets/branding/banner.png') if i==len(rows)-1 else None)
 old=json.loads((p/'branding-report.json').read_text());assert old['output_dimensions']==report['output_dimensions']
 (work/f'report-{i}.json').write_text(json.dumps(report,indent=2));staged.append(out)
 print('STAGED',i+1,[d['action'] for d in report['decisions']],flush=True)
archive_webp(staged,work/'chapter.zip')
with zipfile.ZipFile(work/'chapter.zip') as z:assert len(z.namelist())==7 and z.testzip() is None
backup=work/'before';backup.mkdir();shutil.copy2(job['output_path'],backup/'chapter.zip')
for i,(r,out) in enumerate(zip(rows,staged)):
 p=Path(r['metadata_path']);shutil.copy2(r['output_path'],backup/out.name);shutil.copy2(p/'branding-report.json',backup/f'report-{i}.json')
 shutil.copy2(out,r['output_path']);shutil.copy2(work/f'report-{i}.json',p/'branding-report.json')
 preview=Path(r['output_path']).parent/'previews'/f"{r['id']}.jpg"
 if preview.exists():shutil.move(str(preview),str(backup/preview.name))
shutil.copy2(work/'chapter.zip',job['output_path'])
with Image.open(staged[2]) as im:im.crop((390,10280,800,10600)).save(work/'whoosh-check.png')
print('PUBLISHED 7 WebP quality95 pages; originals backed up',flush=True)
