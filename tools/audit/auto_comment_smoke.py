import json,subprocess,sys,os
os.environ['TYPESET_EXPERIMENTAL_AUTO_COMMENT']='1'
from pathlib import Path
p=Path('artifacts/ch7-bubble-repair')
d=json.loads((p/'translations.json').read_text())
for g in d['groups']:
    g.pop('layout_offset_y',None);g.pop('layout_constraint',None)
(p/'auto-translations.json').write_text(json.dumps(d))
raw=json.loads((p/'pipeline-status.json').read_text())['input']
subprocess.run([sys.executable,'typeset_test.py',str(p/'cleaned.png'),str(p/'auto-translations.json'),'--reference-image',raw,'--surface-map',str(p/'surface-map.json'),'--removal-report',str(p/'removal-report.json'),'--layout-engine','v5','--output',str(p/'auto.png'),'--report',str(p/'auto-report.json')],check=True)
for r in json.loads((p/'auto-report.json').read_text())['results']:
    if r['id'].startswith('bubble-010'):print(r)
