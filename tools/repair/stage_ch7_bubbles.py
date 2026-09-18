import json,shutil,subprocess,sys
from pathlib import Path
from PIL import Image
p=Path('data/jobs/library/erica-when-a-friend-is-obsessed-with-me/chapter-7/381962a1-32d4-40e0-85ae-532069120a16/metadata/page-006')
s=Path('artifacts/ch7-bubble-repair');shutil.copytree(p,s,dirs_exist_ok=True)
d=json.loads((s/'translations.json').read_text());ocr={r['id']:r for r in json.loads((s/'ocr-result.json').read_text())['regions']}
rem=json.loads((s/'removal-report.json').read_text());new=[]
for g in d['groups']:
    if g['id'] not in ('bubble-003','bubble-010'):new.append(g);continue
    cut=3 if g['id']=='bubble-003' else 2
    lines=['GAH! BIKIN KAGET AJA!!','Kamu lagi ngapain di sana?!'] if cut==3 else ['Menyebalkan seperti biasa.','AKU PERGI. LEPAS SENDIRI IKATANMU.']
    oldrem=next(r for r in rem['results'] if r['id']==g['id'])
    for i,ids in enumerate([g['source_region_ids'][:cut],g['source_region_ids'][cut:]]):
        rs=[ocr[k] for k in ids];boxes=[r['bbox'] for r in rs];x=min(b['x'] for b in boxes);y=min(b['y'] for b in boxes)
        ng={**g,'id':g['id']+f'-{i}','source_region_ids':ids,'text':' '.join(r['text'] for r in rs),'translation':lines[i],'bbox':{'x':x,'y':y,'width':max(b['x']+b['width'] for b in boxes)-x,'height':max(b['y']+b['height'] for b in boxes)-y}}
        if (cut==3 and i==1) or (cut==2 and i==0):ng['layout_constraint']='raw_footprint'
        if cut==2 and i==0:ng['layout_offset_y']=-70
        new.append(ng);rem['results'].append({**oldrem,'id':ng['id']})
d['groups']=new
(s/'translations.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));(s/'removal-report.json').write_text(json.dumps(rem,indent=2))
raw=json.loads((s/'pipeline-status.json').read_text())['input']
subprocess.run([sys.executable,'quality_pipeline_v2.py','analyze',raw,str(s/'ocr-result.json'),str(s/'translations.json'),'--output',str(s/'surface-map.json')],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,'typeset_test.py',str(s/'cleaned.png'),str(s/'translations.json'),'--ocr-json',str(s/'ocr-result.json'),'--reference-image',raw,'--surface-map',str(s/'surface-map.json'),'--removal-report',str(s/'removal-report.json'),'--layout-engine','v5','--hard-clip','--min-font-size','22','--output',str(s/'typeset.png'),'--report',str(s/'typeset-report.json')],check=True)
with Image.open(s/'typeset.png') as im:
    im.crop((80,1850,650,2380)).save(s/'spiky.png');im.crop((180,8430,640,8930)).save(s/'oval.png')
print('STAGED ONLY')
