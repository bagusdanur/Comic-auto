"""User-approved removal of the two chapter-5 promotional balloons."""
import json
import shutil
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image
from enclosed_bubble import enclosed_bubble_mask
from webp_archive import archive_webp

jid='4a075ced-c16a-4d6f-97ba-464db33ef501'
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
job=c.execute('select * from jobs where id=?',(jid,)).fetchone()
assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall()
backup=Path('artifacts/ch5-promo-blank');backup.mkdir(exist_ok=True)
prepared=[]
for num,ids in [(2,{'bubble-003','bubble-004'}),(5,{'bubble-004','bubble-005'})]:
    row=rows[num-1];p=Path(row['metadata_path']);out=Path(row['output_path'])
    work=backup/p.name;work.mkdir(exist_ok=True)
    assert not (work/'before-output.png').exists(), 'Already prepared; inspect before rerunning'
    translations=json.loads((p/'translations.json').read_text())
    selected=[g for g in translations['groups'] if g['id'] in ids]
    assert len(selected)==2 and 'BROUGHT TO YOU' in selected[0]['text']
    boxes=[g['bbox'] for g in selected];x=min(b['x'] for b in boxes);y=min(b['y'] for b in boxes)
    right=max(b['x']+b['width'] for b in boxes);bottom=max(b['y']+b['height'] for b in boxes)
    src=json.loads((p/'pipeline-status.json').read_text())['input']
    raw=np.array(Image.open(src).convert('RGB'))
    found=enclosed_bubble_mask(raw,dict(x=x,y=y,width=right-x,height=bottom-y),[])
    assert found is not None, 'Cannot safely identify balloon'
    bounds,mask=found;x1,y1,x2,y2=bounds
    assert x1>80 and x2<740 and y-200<y1<y and bottom<y2<bottom+230, bounds
    report=json.loads((p/'branding-report.json').read_text());assert not report.get('source_header_trimmed_at')
    targets=[out]+[p/name for name in ('typeset.png','cleaned.png') if (p/name).exists()]
    for i,target in enumerate(targets):
        a=np.array(Image.open(target).convert('RGB'));assert a.shape==raw.shape
        shutil.copy2(target,work/('before-output.png' if i==0 else 'before-'+target.name))
        local=a[y1:y2,x1:x2];local[mask>0]=255
        Image.fromarray(a).save(work/('after-output.png' if i==0 else 'after-'+target.name))
    Image.open(work/'after-output.png').crop((80,y1-30,740,y2+30)).save(work/'preview.png')
    regs=[dict(r) for r in c.execute('select * from regions where image_id=?',(row['id'],))]
    (work/'regions-before.json').write_text(json.dumps(regs,indent=2))
    deleted=[]
    for r in regs:
        b=json.loads(r['bbox_json']);cx=b['x']+b['width']/2;cy=b['y']+b['height']/2
        if x1<cx<x2 and y1<cy<y2:deleted.append(r['id'])
    for name in ('translations.json','typeset-report.json'):
        shutil.copy2(p/name,work/('before-'+name))
    translations['groups']=[g for g in translations['groups'] if g['id'] not in ids]
    translations.setdefault('excluded_promotions',[]).extend(selected)
    typeset=json.loads((p/'typeset-report.json').read_text())
    typeset['results']=[g for g in typeset['results'] if g['id'] not in ids]
    prepared.append((row,p,work,targets,deleted,translations,typeset))
    print('PREPARED',num,bounds,'regions',len(deleted),flush=True)

shutil.copy2(job['output_path'],backup/'before-chapter.zip')
for row,p,work,targets,deleted,translations,typeset in prepared:
    for i,target in enumerate(targets):shutil.copy2(work/('after-output.png' if i==0 else 'after-'+target.name),target)
    (p/'translations.json').write_text(json.dumps(translations,indent=2))
    (p/'typeset-report.json').write_text(json.dumps(typeset,indent=2))
    for rid in deleted:c.execute('delete from regions where id=? and image_id=?',(rid,row['id']))
    c.execute('update images set region_count=(select count(*) from regions where image_id=?) where id=?',(row['id'],row['id']))
    preview=Path(row['output_path']).parent/'previews'/f"{row['id']}.jpg"
    if preview.exists():shutil.move(str(preview),str(work/'before-preview.jpg'))
c.commit()
print('ARCHIVED',archive_webp([Path(r['output_path']) for r in rows],Path(job['output_path'])),flush=True)
