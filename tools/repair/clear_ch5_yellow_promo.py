"""Remove two user-confirmed yellow Thunder promotional text blocks."""
import json,shutil,sqlite3
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from xianscan_client import XianScanClient
from webp_archive import archive_webp

jid='4a075ced-c16a-4d6f-97ba-464db33ef501'
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
job=c.execute('select * from jobs where id=?',(jid,)).fetchone();assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall()
backup=Path('artifacts/ch5-yellow-promo');backup.mkdir(exist_ok=True)
client=XianScanClient('http://127.0.0.1:8123');prepared=[]
for num,gid,start,end in [(3,'bubble-003',5481,5633),(6,'bubble-006',6513,6663)]:
    row=rows[num-1];p=Path(row['metadata_path']);out=Path(row['output_path']);work=backup/p.name;work.mkdir(exist_ok=True)
    assert not (work/'before-output.png').exists()
    trans=json.loads((p/'translations.json').read_text());group=next(g for g in trans['groups'] if g['id']==gid)
    assert 'EXCLUSIVELY ON OUR WEBSITE' in group['text']
    assert not json.loads((p/'branding-report.json').read_text()).get('source_header_trimmed_at')
    raw=np.array(Image.open(json.loads((p/'pipeline-status.json').read_text())['input']).convert('RGB'))
    output=np.array(Image.open(out).convert('RGB'));assert output.shape==raw.shape
    y1=start-420;y2=end+420
    context=raw[y1:y2].copy();mask=np.zeros(context.shape[:2],np.uint8)
    for source in (raw,output):
        local=source[y1:y2];yellow=(local[:,:,0]>155)&(local[:,:,1]>125)&(local[:,:,2]<135)&(local[:,:,0].astype(int)-local[:,:,2]>70)
        allowed=np.zeros_like(yellow);allowed[start-y1-10:end-y1+10,55:744]=True
        mask[yellow&allowed]=255
    mask=cv2.dilate(mask,np.ones((13,13),np.uint8))
    assert 5000<np.count_nonzero(mask)<130000
    try:repaired=client.inpaint_mask(context,mask)
    except Exception as exc:
        if getattr(exc,'response',None) is not None:print(exc.response.text[:1000],flush=True)
        raise
    assert repaired.shape==context.shape and not np.any(np.any(repaired!=context,axis=2)&(mask==0))
    targets=[out]+[p/name for name in ('typeset.png','cleaned.png') if (p/name).exists()]
    for i,target in enumerate(targets):
        a=np.array(Image.open(target).convert('RGB'));shutil.copy2(target,work/('before-output.png' if i==0 else 'before-'+target.name))
        a[y1:y2][mask>0]=repaired[mask>0]
        Image.fromarray(a).save(work/('after-output.png' if i==0 else 'after-'+target.name))
    Image.open(work/'after-output.png').crop((0,y1,800,y2)).save(work/'preview.png')
    regs=[dict(r) for r in c.execute('select * from regions where image_id=?',(row['id'],))];(work/'regions-before.json').write_text(json.dumps(regs,indent=2))
    deleted=[]
    for r in regs:
        b=json.loads(r['bbox_json']);cy=b['y']+b['height']/2
        if start-20<cy<end+20:deleted.append(r['id'])
    for name in ('translations.json','typeset-report.json'):shutil.copy2(p/name,work/('before-'+name))
    trans['groups']=[g for g in trans['groups'] if g['id']!=gid];trans.setdefault('excluded_promotions',[]).append(group)
    report=json.loads((p/'typeset-report.json').read_text());report['results']=[g for g in report['results'] if g['id']!=gid]
    prepared.append((row,p,work,targets,deleted,trans,report))
    print('PREPARED',num,flush=True)
shutil.copy2(job['output_path'],backup/'before-chapter.zip')
for row,p,work,targets,deleted,trans,report in prepared:
    for i,target in enumerate(targets):shutil.copy2(work/('after-output.png' if i==0 else 'after-'+target.name),target)
    (p/'translations.json').write_text(json.dumps(trans,indent=2));(p/'typeset-report.json').write_text(json.dumps(report,indent=2))
    for rid in deleted:c.execute('delete from regions where id=? and image_id=?',(rid,row['id']))
    c.execute('update images set region_count=(select count(*) from regions where image_id=?) where id=?',(row['id'],row['id']))
    preview=Path(row['output_path']).parent/'previews'/f"{row['id']}.jpg"
    if preview.exists():shutil.move(str(preview),str(work/'before-preview.jpg'))
c.commit()
print('ARCHIVED',archive_webp([Path(r['output_path']) for r in rows],Path(job['output_path'])),flush=True)
