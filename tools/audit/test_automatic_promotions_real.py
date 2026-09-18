"""Replay original chapter-5 artifacts in isolated outputs, not production jobs."""
import json,sqlite3,shutil
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from typeset_engine_v5 import render_page
c=sqlite3.connect('data/translator.sqlite3')
root=Path(c.execute("select input_path from jobs where id='4a075ced-c16a-4d6f-97ba-464db33ef501'").fetchone()[0]).parent.parent
for num,backup in [(2,'ch5-promo-blank'),(3,'ch5-yellow-promo'),(5,'ch5-promo-blank'),(6,'ch5-yellow-promo')]:
    page=root/'metadata'/f'page-{num:03d}';work=Path('artifacts/automatic-promo-regression')/page.name;work.mkdir(parents=True,exist_ok=True)
    source=json.loads((page/'pipeline-status.json').read_text())['input']
    old=Path('artifacts')/backup/page.name
    for name in ('ocr-result.json','surface-map.json','removal-report.json','pipeline-status.json'):shutil.copy2(page/name,work/name)
    shutil.copy2(old/'before-translations.json',work/'translations.json')
    shutil.copy2(old/'before-cleaned.png',work/'cleaned.png')
    args=SimpleNamespace(reference_image=source,cleaned_image=str(work/'cleaned.png'),translations_json=str(work/'translations.json'),ocr_json=str(work/'ocr-result.json'),surface_map=str(work/'surface-map.json'),removal_report=str(work/'removal-report.json'),font='assets/fonts/wildwords/CC Wild Words Roman.ttf',max_font_size=36,min_font_size=22,output=str(work/'typeset.png'),report=str(work/'typeset-report.json'))
    render_page(args)
    report=json.loads((work/'typeset-report.json').read_text());removed=[d for d in report['automatic_promotions'] if d['action']=='REMOVED_PROMOTION']
    assert len(removed)==1,(num,report['automatic_promotions'])
    assert all(r['status']=='REMOVED_PROMO' for r in report['results'] if r['id'] in removed[0]['group_ids'])
    b=removed[0]['bbox'];im=Image.open(work/'typeset.png');im.crop((0,b['y']-150,im.width,b['y']+b['height']+180)).save(work/'preview.png')
    print('REAL_PASS',num,removed[0]['kind'],removed[0]['group_ids'],flush=True)
