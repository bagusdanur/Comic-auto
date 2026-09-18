import collections
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

stage=Path(sys.argv[1]);counts=collections.Counter();unexpected=0
for page in sorted(stage.glob('page-*')):
    report=json.loads((page/'branding-report.json').read_text())
    assert report['branding_engine'] in ('v5.2','v5.3')
    source=np.array(Image.open(page/'typeset.png').convert('RGB'))
    dest=next(f for f in page.glob('*.png') if f.name not in ('typeset.png','cleaned.png','mask-debug.png'))
    output=np.array(Image.open(dest).convert('RGB'))
    top=report.get('source_header_trimmed_at') or 0
    bottom=report.get('source_footer_trimmed_at') or source.shape[0]
    retained=source[top:bottom]
    assert output.shape[1]==retained.shape[1] and output.shape[0]>=retained.shape[0]
    allowed=np.zeros(source.shape[:2],bool)
    for decision in report['decisions']:
        x,y,x2,y2=decision['context'];allowed[y:y2,x:x2]=True
        counts[decision['action']]+=1
    for replacement in report['replacements']:
        x,y,width,height=replacement['logo'];y+=top
        allowed[y:y+height,x:x+width]=True
    changed=np.any(output[:len(retained)]!=retained,axis=2)
    unexpected+=int(np.count_nonzero(changed&~allowed[top:bottom]))
    if top:counts['CROPPED_WARNING_HEADER']+=1
    if bottom<len(source):counts['CROPPED_SOURCE_FOOTER']+=1
    if report['banner']:counts['RYUKOMIK_BANNER']+=1
assert unexpected==0,unexpected
print(json.dumps({'pages':len(list(stage.glob('page-*'))),'actions':dict(counts),'unexpected_changed_pixels':unexpected}))
