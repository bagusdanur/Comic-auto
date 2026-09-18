from pathlib import Path
import json
from PIL import Image
root=Path('data/jobs/library/erica-when-a-friend-is-obsessed-with-me/chapter-2/3c9b4659-1541-4a2d-962d-55ac2829cb15/metadata')
dest=Path('artifacts/v52-audit');dest.mkdir(exist_ok=True)
for name,box in [('page-001',(0,0,800,1600)),('page-002',(100,1330,450,1540)),('page-025',(0,1180,800,2047))]:
    p=root/name
    source=json.loads((p/'pipeline-status.json').read_text())['input']
    with Image.open(source) as im: im.crop(box).save(dest/(name+'.png'))
