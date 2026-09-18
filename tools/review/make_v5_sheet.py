import json
from pathlib import Path
from PIL import Image,ImageDraw
import sys

stage=Path(sys.argv[1])
rows=[]
for page in sorted(stage.glob('page-*')):
    if not (page/'typeset-report.json').exists(): continue
    report=json.loads((page/'typeset-report.json').read_text())
    output=Image.open(page/'typeset.png').convert('RGB')
    for r in report['results']:
        if r['status']!='TYPESET':continue
        b=r['bbox'];x,y,w,h=(b[k] for k in ('x','y','width','height'))
        crop=output.crop((max(0,x-12),max(0,y-12),min(output.width,x+w+12),min(output.height,y+h+12)))
        crop.thumbnail((450,270))
        rows.append((f"{page.name} {r['id']} {r['font_size']}px",crop))
for batch in range(0,len(rows),12):
    sheet=Image.new('RGB',(920,1800),'#252525');draw=ImageDraw.Draw(sheet)
    for i,(label,crop) in enumerate(rows[batch:batch+12]):
        x=(i%2)*460;y=(i//2)*300
        draw.text((x+4,y+4),label,fill='white');sheet.paste(crop,(x+4,y+25))
    sheet.save(stage/f'sheet-{batch//12+1}.jpg')
print('blocks',len(rows),'sheets',(len(rows)+11)//12)
