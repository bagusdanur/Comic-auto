import json,sqlite3,shutil
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from typeset_engine_v5 import render_page
from brand_postprocess import apply_branding
from priority_seams import prepare_inputs
root=Path(sqlite3.connect('data/translator.sqlite3').execute("select input_path from jobs where id='adbfe8cc-304c-4fdc-8287-40e6ac878345'").fetchone()[0]).parent.parent
for num in (12,2):
 p=root/'metadata'/f'page-{num:03d}';stage=Path('artifacts/priority-engine-shadow')/p.name;stage.mkdir(parents=True,exist_ok=True)
 for name in ('cleaned.png','ocr-result.json','translations.json','removal-report.json','surface-map.json','pipeline-status.json'):shutil.copy2(p/name,stage/name)
 src=json.loads((p/'pipeline-status.json').read_text())['input']
 args=SimpleNamespace(reference_image=src,cleaned_image=str(stage/'cleaned.png'),translations_json=str(stage/'translations.json'),ocr_json=str(stage/'ocr-result.json'),surface_map=str(stage/'surface-map.json'),removal_report=str(stage/'removal-report.json'),font='assets/fonts/wildwords/CC Wild Words Roman.ttf',max_font_size=36,min_font_size=22,output=str(stage/'typeset.png'),report=str(stage/'typeset-report.json'))
 render_page(args)
 if num==12:
  d=json.loads((stage/'typeset-report.json').read_text());print('BUBBLE',d['results'][0],flush=True)
  Image.open(stage/'typeset.png').crop((0,400,800,1100)).save(stage/'preview.png')
 else:
  report=apply_branding(stage/'typeset.png',stage/'ocr-result.json',Path('assets/branding/watermark.png'),Path('assets/branding/watermark-text.png'))
  print('BRAND',report,flush=True);assert report['source_header_trimmed_at']==481
  assert any(d['action']=='REMOVED_VORTEX_EMBLEM' for d in report['decisions'])
  Image.open(stage/'typeset.png').crop((250,5850,550,6250)).save(stage/'preview.png')
paths=prepare_inputs(sorted((root/'original').iterdir()),root/'prepared-seams-v1')
print('SEAMS',json.loads((root/'prepared-seams-v1/manifest.json').read_text())['boundary_shifts'],flush=True)
assert sum(Image.open(p).height for p in paths)==sum(Image.open(p).height for p in (root/'original').iterdir())
