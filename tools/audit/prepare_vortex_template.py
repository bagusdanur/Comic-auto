import sqlite3,json
from pathlib import Path
import numpy as np
from PIL import Image
from vortex_emblems import red_mask
root=Path(sqlite3.connect('data/translator.sqlite3').execute("select input_path from jobs where id='adbfe8cc-304c-4fdc-8287-40e6ac878345'").fetchone()[0]).parent.parent
src=json.loads((root/'metadata/page-002/pipeline-status.json').read_text())['input']
target=Path('assets/branding/vortex-red-template.png');target.parent.mkdir(parents=True,exist_ok=True)
assert not target.exists()
Image.fromarray(red_mask(np.array(Image.open(src).convert('RGB'))[0:154,18:161])).save(target)
