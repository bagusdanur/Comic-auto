"""Render a reproducible red-RAW regression sample using the production engine."""
import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image,ImageDraw,ImageFont
from typeset_engine_v5 import render_page

p=Path('artifacts/v53-color-test');p.mkdir(exist_ok=True)
font='assets/fonts/wildwords/CC Wild Words Roman.ttf'
raw=Image.new('RGB',(800,320),'white');draw=ImageDraw.Draw(raw)
draw.ellipse((190,30,610,285),outline='black',width=4)
clean=raw.copy();clean.save(p/'cleaned.png')
draw.text((270,120),'STOP RIGHT THERE!',font=ImageFont.truetype(font,28),fill=(220,20,35))
raw.save(p/'raw.png')
b={'x':265,'y':118,'width':285,'height':42}
(p/'translations.json').write_text(json.dumps({'groups':[{'id':'bubble-001','text':'STOP RIGHT THERE!','translation':'BERHENTI DI SITU!','bbox':b}]}))
(p/'surface-map.json').write_text(json.dumps({'surfaces':[{'id':'bubble-001','surface_type':'BUBBLE_FLAT'}]}))
(p/'removal-report.json').write_text(json.dumps({'results':[{'id':'bubble-001','status':'INPAINTED'}]}))
render_page(SimpleNamespace(reference_image=p/'raw.png',cleaned_image=p/'cleaned.png',translations_json=p/'translations.json',surface_map=p/'surface-map.json',removal_report=p/'removal-report.json',max_font_size=48,min_font_size=22,font=font,output=p/'typeset.png',report=p/'typeset-report.json'))
report=json.loads((p/'typeset-report.json').read_text());color=report['results'][0]['color']
assert color[0]>180 and color[1]<70 and color[2]<80,color
sheet=Image.new('RGB',(800,640),'white');sheet.paste(raw,(0,0))
with Image.open(p/'typeset.png') as im:sheet.paste(im,(0,320))
sheet.save(p/'comparison.png');print(report['results'][0])
