import json,subprocess,sys,os
from pathlib import Path
os.environ['TYPESET_EXPERIMENTAL_AUTO_COMMENT']='1'
root=Path('data/jobs/library/erica-when-a-friend-is-obsessed-with-me')
cases=[(4,'6683eec6-57f9-4283-b83d-9d7911b7233e',7),(6,'a89d5e21-d844-40cf-a0f2-72e2389aebdf',7),(7,'381962a1-32d4-40e0-85ae-532069120a16',3)]
out=Path('artifacts/auto-comment-regression');out.mkdir(exist_ok=True)
for chapter,jid,page in cases:
    p=root/f'chapter-{chapter}'/jid/'metadata'/f'page-{page:03d}'
    raw=json.loads((p/'pipeline-status.json').read_text())['input']
    dest=out/f'ch{chapter}-page{page}'
    subprocess.run([sys.executable,'typeset_test.py',str(p/'cleaned.png'),str(p/'translations.json'),'--reference-image',raw,'--surface-map',str(p/'surface-map.json'),'--removal-report',str(p/'removal-report.json'),'--layout-engine','v5','--output',str(dest.with_suffix('.png')),'--report',str(dest.with_suffix('.json'))],check=True)
    report=json.loads(dest.with_suffix('.json').read_text())
    assert all(r.get('overflow_pixels',0)==0 for r in report['results'])
    print(chapter,page,'PASS',[(r['id'],r['bbox']) for r in report['results'] if r.get('placement_strategy')=='automatic-exterior-comment-v1'],flush=True)
