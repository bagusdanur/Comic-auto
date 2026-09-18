"""Render V5 candidates in an isolated directory; never mutate live job results."""
import json
import sqlite3
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from typeset_engine_v5 import fit_text_v5, source_surface_mask

root = Path.cwd()
db = sqlite3.connect('file:data/translator.sqlite3?mode=ro', uri=True)
job_id, archive = db.execute('SELECT id,output_path FROM jobs ORDER BY created_at DESC LIMIT 1').fetchone()
job = Path(archive).parent.parent
dest = root/'artifacts'/'v5-shadow'/job_id
dest.mkdir(parents=True, exist_ok=True)
font = root/'assets/fonts/wildwords/CC Wild Words Roman.ttf'
summary=[]
rows=[]
for page in sorted((job/'metadata').glob('page-*')):
    pipeline=json.loads((page/'pipeline-status.json').read_text())
    groups=json.loads((page/'translations.json').read_text())['groups']
    if not groups:
        Image.open(pipeline['input']).save(dest/(page.name+'.png'))
        summary.append({'page':page.name,'results':[],'passthrough':True})
        continue
    surfaces={s['id']:s for s in json.loads((page/'surface-map.json').read_text())['surfaces']}
    old={r['id']:r for r in json.loads((page/'typeset-report.json').read_text())['results']}
    ref=np.array(Image.open(pipeline['input']).convert('RGB'))
    output=Image.open(page/'cleaned.png').convert('RGB')
    draw=ImageDraw.Draw(output)
    results=[]
    for group in groups:
        previous=old.get(group['id'],{})
        if previous.get('status')=='PRESERVED':
            b=group['bbox']; x,y,w,h=(b[k] for k in ('x','y','width','height'))
            output.paste(Image.fromarray(ref[y:y+h,x:x+w]),(x,y))
            results.append({'id':group['id'],'status':'PRESERVED'})
            continue
        surface=surfaces.get(group['id'],{})
        bounds,mask=source_surface_mask(ref,group['bbox'],surface,[g['bbox'] for g in groups if g['id']!=group['id']])
        flat=surface.get('surface_type')=='BUBBLE_FLAT'
        stroke=0 if flat else 2
        preferred=max(28,min(42,int(previous.get('preferred_font_size') or 32)))
        fitted=fit_text_v5(group['translation'],font,mask,minimum=22,maximum=preferred,stroke=stroke)
        result={'id':group['id'],'status':'TYPESET' if fitted else 'FAILED_LAYOUT','bounds':bounds,'previous_font':previous.get('font_size')}
        if fitted:
            ox,oy=fitted['origin']
            draw.multiline_text((bounds[0]+ox,bounds[1]+oy),fitted['wrapped'],font=fitted['font'],spacing=fitted['spacing'],align='center',fill='black',stroke_width=stroke,stroke_fill='white')
            result.update({k:fitted[k] for k in ('font_size','wrapped','line_count','overflow_pixels')})
        results.append(result)
        if previous.get('status')=='NEEDS_REVIEW' or not fitted:
            x1,y1,x2,y2=bounds
            rows.append((page.name+' '+group['id']+' '+result['status'],Image.fromarray(ref[y1:y2,x1:x2]),output.crop(bounds)))
    output.save(dest/(page.name+'.png'))
    summary.append({'page':page.name,'results':results})
    print(page.name,[(r['id'],r['status']) for r in results if r['status']=='FAILED_LAYOUT'],flush=True)
(dest/'report.json').write_text(json.dumps(summary,indent=2))
sheet=Image.new('RGB',(1000,sum(max(a.height,b.height)+30 for _,a,b in rows) or 1),'#222222')
y=0
for label,a,b in rows:
    ImageDraw.Draw(sheet).text((5,y),label,fill='white')
    sheet.paste(a,(0,y+25));sheet.paste(b,(500,y+25));y+=max(a.height,b.height)+30
sheet.save(dest/'review.png')
print('REPORT',dest,flush=True)
