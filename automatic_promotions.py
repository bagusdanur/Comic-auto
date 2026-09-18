"""Conservative OCR-anchored removal of Thunder's embedded promotions."""
import re
import numpy as np
from enclosed_bubble import enclosed_bubble_mask


def normalized(text):
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())


def detect_promotions(regions, width):
    blocks=[];seen=set()
    for anchor in regions:
        if 'ENTHUNDERSCANSCOM' not in normalized(anchor.get('text','')):continue
        ay=anchor['bbox']['y']
        nearby=sorted([r for r in regions if abs(r['bbox']['y']-ay)<width*.24],key=lambda r:r['bbox']['y'])
        eligible=[r for r in nearby if re.search(r'CHAPTER|WEBSITE|PIRATED|CONTINUITY|SUPPORT|THUNDER|READING|EXCLUSIVELY|ENSURE|BROUGHT',str(r.get('text','')),re.I)]
        text=normalized(' '.join(r['text'] for r in eligible))
        if 'THISCHAPTERISBROUGHTTOYOU' in text and 'PIRATEDWEBSITES' in text and 'CONTINUITY' in text:
            kind='PROMOTIONAL_BALLOON'
        elif 'ALLCHAPTERS' in text and 'EXCLUSIVELYONOURWEBSITE' in text:
            kind='PROMOTIONAL_BANNER'
        else:continue
        boxes=[r['bbox'] for r in eligible]
        x=min(b['x'] for b in boxes);y=min(b['y'] for b in boxes)
        right=max(b['x']+b['width'] for b in boxes);bottom=max(b['y']+b['height'] for b in boxes)
        ids={r['id'] for r in eligible}
        # Reject a block containing any unrelated OCR line. Tiny duplicate
        # fragments completely inside confirmed text are allowed.
        unsafe=False
        for r in nearby:
            b=r['bbox'];cx=b['x']+b['width']/2;cy=b['y']+b['height']/2
            if r['id'] not in ids and x<cx<right and y<cy<bottom:
                nested=any(b['x']>=a['x'] and b['y']>=a['y'] and b['x']+b['width']<=a['x']+a['width'] and b['y']+b['height']<=a['y']+a['height'] for a in boxes)
                if len(r.get('text',''))<=8 and nested:ids.add(r['id'])
                else:unsafe=True
        key=(x,y,right,bottom)
        if unsafe or key in seen or bottom-y>width*.3:continue
        seen.add(key);blocks.append({'kind':kind,'bbox':dict(x=x,y=y,width=right-x,height=bottom-y),'source_ids':sorted(ids)})
    return blocks


def remove_promotions(reference, output, regions, groups, inpaint=None):
    """Return modified image, suppressed group IDs and explicit audit records."""
    output=output.copy();suppressed=set();decisions=[]
    h,w=reference.shape[:2]
    for block in detect_promotions(regions,w):
        ids=set(block['source_ids']);related=[g for g in groups if ids.intersection(g.get('source_region_ids',[]))]
        if any(not set(g.get('source_region_ids',[])).issubset(ids) for g in related):continue
        b=block['bbox'];x,y,bw,bh=(int(b[k]) for k in ('x','y','width','height'))
        others=[r['bbox'] for r in regions if r['id'] not in ids]
        if block['kind']=='PROMOTIONAL_BALLOON':
            found=enclosed_bubble_mask(reference,b,[])
            if found is None:continue
            bounds,mask=found;a,t,c,d=bounds
            if any(a<r['x']+r['width']/2<c and t<r['y']+r['height']/2<d for r in others):continue
            output[t:d,a:c][mask>0]=255
        else:
            a=max(0,x-14);c=min(w,x+bw+16);t=max(0,y-12);d=min(h,y+bh+14)
            if any(r['x']<c and r['x']+r['width']>a and r['y']<d and r['y']+r['height']>t for r in others):continue
            top=max(0,t-500);bottom=min(h,d+500)
            context=reference[top:bottom].copy();mask=np.zeros(context.shape[:2],np.uint8);mask[t-top:d-top,a:c]=255
            if inpaint is None:
                from xianscan_client import XianScanClient
                import os
                inpaint=XianScanClient(os.getenv('XIANSCAN_URL','http://127.0.0.1:8123')).inpaint_mask
            try:repaired=inpaint(context,mask)
            except Exception as exc:
                decisions.append({**block,'action':'PRESERVED_PROMOTION','reason':type(exc).__name__})
                continue
            if repaired.shape!=context.shape or np.any(np.any(repaired!=context,axis=2)&(mask==0)):
                decisions.append({**block,'action':'PRESERVED_PROMOTION','reason':'UNSAFE_INPAINT_RESULT'})
                continue
            output[top:bottom][mask>0]=repaired[mask>0]
        suppressed.update(g['id'] for g in related)
        # Persist the exact rectangle whose pixels the remover was allowed to
        # modify. Downstream QC can then distinguish this intentional cleanup
        # from an accidental edit elsewhere in the artwork.
        decisions.append({
            **block,
            'action':'REMOVED_PROMOTION',
            'affected_bbox':{'x':a,'y':t,'width':c-a,'height':d-t},
            'group_ids':[g['id'] for g in related],
        })
    return output,suppressed,decisions
