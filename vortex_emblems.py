"""Template-confirmed Vortex emblem cleanup with dialogue collision guards."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from branding_ink_guard import intersects_protected

def red_mask(a):
    h=cv2.cvtColor(a,cv2.COLOR_RGB2HSV)
    return (((h[:,:,0]<12)|(h[:,:,0]>170))&(h[:,:,1]>85)&(h[:,:,2]>45)).astype(np.uint8)*255

def find_emblems(reference,template):
    binary=red_mask(reference);w=reference.shape[1];candidates=[]
    if not np.any(binary):return []
    for base in range(76,101,4):
        width=max(12,round(base*w/800));height=round(width*template.shape[0]/template.shape[1])
        if height>binary.shape[0] or width>w:continue
        t=cv2.resize(template,(width,height))
        scores=cv2.matchTemplate(binary,t,cv2.TM_CCOEFF_NORMED)
        yy,xx=np.where(scores>.66)
        candidates.extend((float(scores[y,x]),int(x),int(y),width,height) for y,x in zip(yy,xx))
    selected=[]
    for c in sorted(candidates,reverse=True):
        if any(abs(c[1]-s[1])<c[3] and abs(c[2]-s[2])<c[4] for s in selected):continue
        selected.append(c)
    return selected

def clean_emblems(reference,output,protected,top,bottom,inpaint):
    path=Path(__file__).parent/'assets/branding/vortex-red-template.png'
    if not path.exists():return output,[{'action':'EMBLEM_TEMPLATE_UNAVAILABLE'}]
    template=np.array(Image.open(path).convert('L'));h,w=reference.shape[:2];decisions=[]
    for score,x,y,bw,bh in find_emblems(reference,template):
        x1=max(0,x-6);x2=min(w,x+bw+6);y1=max(0,y-6);y2=min(h,y+bh+6)
        if y1<top or y2>bottom:continue
        mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=255
        def actual_collision(r):
            box=r['bbox'];area=box['width']*box['height']
            # Very low-confidence OCR sometimes puts a huge SFX rectangle
            # across the adjacent emblem. It is not a measured glyph mask.
            if r.get('confidence',1)<.35 and area>6*bw*bh and '_ink_mask' not in r:
                return False
            return intersects_protected(mask,(0,0),r)
        if any(actual_collision(r) for r in protected):
            decisions.append({'action':'PRESERVED_EMBLEM_DIALOGUE_OVERLAP','score':score,'bbox':[x1,y1,x2,y2]});continue
        a=max(0,x1-180);b=max(0,y1-180);c=min(w,x2+180);d=min(h,y2+180)
        context=reference[b:d,a:c];local=mask[b:d,a:c]
        try:repaired=inpaint(context,local)
        except Exception as exc:
            decisions.append({'action':'PRESERVED_EMBLEM_INPAINT_ERROR','reason':type(exc).__name__});continue
        if repaired.shape!=context.shape or np.any(np.any(repaired!=context,axis=2)&(local==0)):
            decisions.append({'action':'PRESERVED_UNSAFE_EMBLEM_RESULT'});continue
        output[b:d,a:c][local>0]=repaired[local>0]
        decisions.append({'action':'REMOVED_VORTEX_EMBLEM','score':score,'bbox':[x1,y1,x2,y2],'outside_mask_changed_pixels':0})
    return output,decisions
