from pathlib import Path
import json
import shutil
import numpy as np
from PIL import Image
from brand_postprocess import apply_branding

page=Path('data/jobs/library/erica-when-a-friend-is-obsessed-with-me/chapter-4/6683eec6-57f9-4283-b83d-9d7911b7233e/metadata/page-007')
work=Path('artifacts/ch4-branding-fix');work.mkdir(exist_ok=True)
dest=work/'page-007.png';shutil.copy2(page/'typeset.png',dest)
report=apply_branding(dest,page/'ocr-result.json',Path('assets/branding/watermark.png'),Path('assets/branding/watermark-text.png'))
(work/'branding-report.json').write_text(json.dumps(report,indent=2))
before=np.array(Image.open(page/'typeset.png').convert('RGB'))
after=np.array(Image.open(dest).convert('RGB'))
assert before.shape==after.shape
allowed=np.zeros(before.shape[:2],bool)
for decision in report['decisions']:
 x,y,x2,y2=decision['context'];allowed[y:y2,x:x2]=True
for replacement in report['replacements']:
 x,y,w,h=replacement['logo'];allowed[y:y+h,x:x+w]=True
assert not np.any(np.any(before!=after,axis=2)&~allowed)
Image.fromarray(after[1400:1940,0:800]).save(work/'checked-crop.png')
print(json.dumps({'collision_guard':report['collision_guard'],'replacements':report['replacements'],'unexpected_changed_pixels':0}))
