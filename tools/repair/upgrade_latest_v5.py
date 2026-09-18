"""Stage V5 cleanup/layout/QC for a completed job, then optionally publish."""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from PIL import Image
from quality_pipeline_v2 import validate
from brand_postprocess import apply_branding
from webp_archive import archive_webp
from branding_coordinates import project_region

p=argparse.ArgumentParser()
p.add_argument('job_id')
p.add_argument('--publish',action='store_true')
p.add_argument('--publish-only',action='store_true')
p.add_argument('--rebrand-only',action='store_true')
p.add_argument('--branding-v51',action='store_true')
p.add_argument('--branding-v52',action='store_true')
p.add_argument('--branding-v53',action='store_true')
p.add_argument('--rerender-only',action='store_true')
p.add_argument('--refresh-pages',nargs='+')
p.add_argument('--resume-staging',action='store_true')
args=p.parse_args()
if args.branding_v51:
    from branding_v51 import apply_branding_v51 as apply_branding
if args.branding_v52:
    from branding_v52 import apply_branding_v52 as apply_branding
if args.branding_v53:
    from branding_v53 import apply_branding_v53 as apply_branding
root=Path.cwd()
db=sqlite3.connect('data/translator.sqlite3',timeout=30)
db.row_factory=sqlite3.Row
job=db.execute('SELECT * FROM jobs WHERE id=?',(args.job_id,)).fetchone()
if not job or job['status'] not in ('COMPLETED','REVIEW'):
    raise SystemExit('Expected a finished job')
jobroot=Path(job['output_path']).parent.parent
stage=root/'artifacts'/('v53-release' if args.branding_v53 else 'v52-release' if args.branding_v52 else 'v51-release' if args.branding_v51 else 'v5-release')/args.job_id
stage.mkdir(parents=True,exist_ok=True)
pages=sorted((jobroot/'metadata').glob('page-*'))
outputs=sorted((jobroot/'output').glob('*.png'))
if len(pages)!=len(outputs): raise SystemExit('Page/output mismatch')
staged=[]
for index,(page,output) in enumerate(zip(pages,outputs)):
    work=stage/page.name
    work.mkdir(exist_ok=True)
    if args.refresh_pages and page.name not in args.refresh_pages:
        ready=json.loads((work/'pipeline-status.json').read_text())
        if ready.get('branding_engine')!='v5.3' or ready.get('status')!='COMPLETED':
            raise RuntimeError('Cannot reuse incomplete V5.3 staging')
        dest=work/output.name
        with Image.open(dest) as image:image.verify()
        staged.append((page,work,output,dest))
        continue
    if args.resume_staging and args.branding_v52 and (work/'pipeline-status.json').exists() and (work/output.name).exists():
        ready=json.loads((work/'pipeline-status.json').read_text())
        if ready.get('branding_engine')=='v5.2':
            staged.append((page,work,output,work/output.name))
            print(page.name,'RESUMED',flush=True)
            continue
    if args.publish_only:
        status=json.loads((work/'pipeline-status.json').read_text())
        if status.get('status')!='COMPLETED' or status.get('pipeline_engine')!='v5':
            raise RuntimeError('Staged V5 page is not ready')
        if args.branding_v52 and status.get('branding_engine')!='v5.2':
            raise RuntimeError('Staged V5.2 branding is not ready')
        if args.branding_v53 and status.get('branding_engine')!='v5.3':
            raise RuntimeError('Staged V5.3 branding is not ready')
        dest=work/output.name
        with Image.open(dest) as image: image.verify()
        staged.append((page,work,output,dest))
        continue
    if args.rebrand_only and not (work/'typeset.png').exists():
        for name in ('typeset.png','typeset-report.json','pipeline-status.json'):
            shutil.copy2(page/name,work/name)
    pipeline=json.loads(((work if args.rebrand_only else page)/'pipeline-status.json').read_text())
    original=Path(pipeline['input'])
    groups=json.loads((page/'translations.json').read_text())['groups']
    if groups and not args.rebrand_only:
        cmd=[sys.executable,'remove_text_test.py',str(original),str(page/'ocr-result.json'),str(page/'translations.json'),'--output',str(work/'cleaned.png'),'--mask',str(work/'mask-debug.png'),'--report',str(work/'removal-report.json'),'--surface-map',str(page/'surface-map.json'),'--hybrid-mode','xianscan-v3','--strict-artwork-inpaint']
        if args.rerender_only:
            for name in ('cleaned.png','mask-debug.png','removal-report.json'):
                shutil.copy2(page/name,work/name)
        else:
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        cmd=[sys.executable,'typeset_test.py',str(work/'cleaned.png'),str(page/'translations.json'),'--reference-image',str(original),'--removal-report',str(work/'removal-report.json'),'--surface-map',str(page/'surface-map.json'),'--output',str(work/'typeset.png'),'--report',str(work/'typeset-report.json'),'--layout-engine','v5','--min-font-size','22']
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
        qc=validate(original,work/'cleaned.png',work/'typeset.png',work/'mask-debug.png',page/'surface-map.json',work/'quality-report.json')
        if qc['status']!='PASS': raise RuntimeError(f"{page.name}: {qc['review_reasons']}")
        pipeline.update(quality_v2=qc)
    elif not args.rebrand_only:
        shutil.copy2(page/'typeset.png',work/'typeset.png')
    dest=work/output.name
    shutil.copy2(work/'typeset.png',dest)
    brand=apply_branding(dest,page/'ocr-result.json',root/'assets/branding/watermark.png',root/'assets/branding/watermark-text.png',first_page=index==0,banner_path=root/'assets/branding/banner.png' if index==len(pages)-1 else None)
    (work/'branding-report.json').write_text(json.dumps(brand,indent=2))
    pipeline.update(status='COMPLETED',review_ids=[],pipeline_engine='v5',layout_engine='v5')
    pipeline['branding_engine']=brand.get('branding_engine','legacy')
    (work/'pipeline-status.json').write_text(json.dumps(pipeline,indent=2))
    staged.append((page,work,output,dest))
    print(page.name,'V5 PASS',flush=True)
if not args.publish_only:
    archive_webp([row[3] for row in staged],stage/'chapter-translated.zip')
print('STAGING_COMPLETE',stage,flush=True)
if args.publish or args.publish_only:
    import zipfile
    with zipfile.ZipFile(stage/'chapter-translated.zip') as bundle:
        if len(bundle.namelist())!=len(pages) or bundle.testzip():
            raise RuntimeError('Staged ZIP failed validation')
    current=db.execute('SELECT status FROM jobs WHERE id=?',(args.job_id,)).fetchone()
    if current['status'] not in ('COMPLETED','REVIEW'): raise RuntimeError('Job changed while staging')
    backup=stage/'previous-release'
    backup.mkdir(exist_ok=True)
    if not (backup/'database-regions.json').exists():
        records=[dict(r) for r in db.execute('SELECT regions.* FROM regions JOIN images ON images.id=regions.image_id WHERE images.job_id=?',(args.job_id,)).fetchall()]
        (backup/'database-regions.json').write_text(json.dumps(records,indent=2))
    if not (backup/'chapter-translated.zip').exists():
        shutil.copy2(job['output_path'],backup/'chapter-translated.zip')
    for page,work,output,dest in staged:
        saved=backup/page.name
        saved.mkdir(exist_ok=True)
        for report in page.glob('*.json'):
            if not (saved/report.name).exists(): shutil.copy2(report,saved/report.name)
        for filename in ['cleaned.png','mask-debug.png','removal-report.json','typeset.png','typeset-report.json','quality-report.json','branding-report.json','pipeline-status.json']:
            if (work/filename).exists(): shutil.copy2(work/filename,page/filename)
        shutil.copy2(dest,output)
        row=db.execute('SELECT id FROM images WHERE job_id=? AND metadata_path=?',(args.job_id,str(page))).fetchone()
        if row:
            (output.parent/'previews'/f"{row['id']}.jpg").unlink(missing_ok=True)
            with Image.open(output) as im:
                db.execute("UPDATE images SET status='COMPLETED',width=?,height=? WHERE id=?",(*im.size,row['id']))
            report=json.loads((page/'typeset-report.json').read_text())
            mapped={r['id']:r for r in report['results']}
            branding=json.loads((page/'branding-report.json').read_text())
            source_groups={r['id']:r for r in json.loads((page/'translations.json').read_text())['groups']}
            for region in db.execute('SELECT id,metadata_json FROM regions WHERE image_id=?',(row['id'],)).fetchall():
                old=json.loads(region['metadata_json'] or '{}'); new=mapped.get(old.get('id'))
                if new:
                    projected=project_region(new,source_groups[new['id']]['bbox'],branding)
                    if projected is None:
                        db.execute('DELETE FROM regions WHERE id=?',(region['id'],))
                    else:
                        display_box,new=projected
                        db.execute('UPDATE regions SET status=?,bbox_json=?,metadata_json=? WHERE id=?',(new['status'],json.dumps(display_box),json.dumps(new),region['id']))
            db.execute('UPDATE images SET region_count=(SELECT count(*) FROM regions WHERE image_id=?) WHERE id=?',(row['id'],row['id']))
    shutil.copy2(stage/'chapter-translated.zip',job['output_path'])
    options=json.loads(job['options_json']);options.update(pipeline_engine='v5',layout_engine='v5')
    if args.branding_v51: options['branding_engine']='v5.1'
    if args.branding_v52: options['branding_engine']='v5.2'
    if args.branding_v53: options['branding_engine']='v5.3'
    now=time.time()
    db.execute("UPDATE jobs SET status='COMPLETED',stage='DONE',progress=1,review_count=0,error=NULL,updated_at=?,completed_at=?,options_json=? WHERE id=?",(now,now,json.dumps(options),args.job_id))
    db.commit()
    print('PUBLISHED',args.job_id,flush=True)
