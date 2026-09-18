import json,shutil,sqlite3,uuid
from pathlib import Path
from PIL import Image
from brand_postprocess import apply_branding
from branding_coordinates import project_region
from webp_archive import archive_webp
jid='381962a1-32d4-40e0-85ae-532069120a16'
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
job=c.execute('select * from jobs where id=?',(jid,)).fetchone();assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall();row=rows[5]
s=Path('artifacts/ch7-bubble-repair');p=Path(row['metadata_path']);out=s/'final.png';shutil.copy2(s/'typeset.png',out)
report=apply_branding(out,s/'ocr-result.json',Path('assets/branding/watermark.png'),Path('assets/branding/watermark-text.png'))
(s/'branding-report.json').write_text(json.dumps(report,indent=2))
backup=s/'before-publish';backup.mkdir(exist_ok=False)
shutil.copytree(p,backup/'metadata');shutil.copy2(row['output_path'],backup/'output.png');shutil.copy2(job['output_path'],backup/'chapter.zip')
(backup/'regions.json').write_text(json.dumps([dict(r) for r in c.execute('select * from regions where image_id=?',(row['id'],))],indent=2))
pages=s/'zip-pages';pages.mkdir(exist_ok=True)
for r in rows:shutil.copy2(out if r['id']==row['id'] else r['output_path'],pages/Path(r['output_path']).name)
archive_webp(list(pages.glob('*.png')),s/'chapter.zip')
for name in ['translations.json','surface-map.json','removal-report.json','typeset.png','typeset-report.json','branding-report.json']:shutil.copy2(s/name,p/name)
shutil.copy2(out,row['output_path'])
data=json.loads((s/'translations.json').read_text());results={r['id']:r for r in json.loads((s/'typeset-report.json').read_text())['results']}
c.execute('delete from regions where image_id=?',(row['id'],))
for g in data['groups']:
    projected=project_region(results[g['id']],g['bbox'],report)
    if projected is None:continue
    box,item=projected
    c.execute('insert into regions(id,image_id,source_text,translation,bbox_json,status,metadata_json) values(?,?,?,?,?,?,?)',(str(uuid.uuid4()),row['id'],g['text'],g['translation'],json.dumps(box),item['status'],json.dumps(item)))
c.execute('update images set region_count=(select count(*) from regions where image_id=?) where id=?',(row['id'],row['id']))
preview=Path(row['output_path']).parent/'previews'/f"{row['id']}.jpg"
if preview.exists():shutil.move(str(preview),backup/'preview.jpg')
shutil.copy2(s/'chapter.zip',job['output_path']);c.commit();print('PUBLISHED',jid,'WebP quality95; backed up previous output and regions')
