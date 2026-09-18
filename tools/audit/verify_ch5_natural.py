import json,sqlite3,zipfile
from pathlib import Path
from PIL import Image
root=Path('artifacts/ch5-natural-fix')
page=root/'page-007'
report=json.loads((page/'typeset-report.json').read_text())
b=next(r['bbox'] for r in report['results'] if r['id']=='bubble-013')
with Image.open(page/'final.png') as im:
 im.crop((0,max(0,b['y']-100),im.width,min(im.height,b['y']+b['height']+100))).save(root/'wording-check.png')
with Image.open(root/'page-010/final.png') as im:
 im.save(root/'footer-check.png')
 assert im.size==(800,439)
with zipfile.ZipFile(root/'chapter.zip') as z:
 assert len(z.namelist())==10 and all(n.endswith('.webp') for n in z.namelist())
 assert z.testzip() is None
print('VERIFIED 10 WebP pages, footer banner only')
