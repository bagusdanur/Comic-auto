import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw

stage=Path(sys.argv[1]);rows=[]
for p in sorted(stage.glob('page-*')):
    if not (p/'branding-report.json').exists():continue
    report=json.loads((p/'branding-report.json').read_text())
    outputs=[f for f in p.glob('*.png') if f.name not in ('typeset.png','cleaned.png','mask-debug.png')]
    if not outputs:continue
    with Image.open(outputs[0]) as opened:im=opened.convert('RGB')
    for replacement in report.get('replacements',[]):
        x,y,w,h=replacement['logo']
        crop=im.crop((0,max(0,y-70),im.width,min(im.height,y+h+70)))
        crop.thumbnail((760,260))
        rows.append((p.name+' '+replacement['mode'],crop))
for offset in range(0,len(rows),6):
    sheet=Image.new('RGB',(800,1800),'#252525');draw=ImageDraw.Draw(sheet)
    for i,(label,crop) in enumerate(rows[offset:offset+6]):
        draw.text((10,i*300+8),label,fill='white');sheet.paste(crop,(10,i*300+30))
    sheet.save(stage/f'brand-sheet-{offset//6+1}.jpg')
print('placement_crops',len(rows))
