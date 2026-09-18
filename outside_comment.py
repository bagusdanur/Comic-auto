"""Bounded placement candidates for comments beside independently detected balloons."""
import cv2
import numpy as np
from enclosed_bubble import enclosed_bubble_mask
from open_bubble import open_bubble_mask

def comment_candidates(reference, cleaned, box, others):
    # Interior dialogue stays with its own balloon, not this exterior solver.
    x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
    own=enclosed_bubble_mask(reference,box,[])
    if own is None:own=open_bubble_mask(reference,box,others)
    if own is not None:
        (a,b,c,d),mask=own
        l,t,r,bt=max(a,x),max(b,y),min(c,x+w),min(d,y+h)
        if r>l and bt>t and np.count_nonzero(mask[t-b:bt-b,l-a:r-a])>w*h*.8:return []
    ph,pw=reference.shape[:2];radius=min(100,max(32,h))
    left,top,right,bottom=max(0,x-radius-12),max(0,y-radius-12),min(pw,x+w+radius+12),min(ph,y+h+radius+12)
    blocked=np.zeros((bottom-top,right-left),np.uint8);found=False
    for other in others:
        ox,oy,ow,oh=(int(other[k]) for k in ('x','y','width','height'))
        if ox+ow<left-150 or ox>right+150 or oy+oh<top-150 or oy>bottom+150:continue
        balloon=enclosed_bubble_mask(reference,other,[])
        if balloon is None:balloon=open_bubble_mask(reference,other,[box,*others])
        if balloon is None:continue
        (a,b,c,d),mask=balloon
        l,t,r,bt=max(a,left),max(b,top),min(c,right),min(d,bottom)
        if r<=l or bt<=t:continue
        found=True
        blocked[t-top:bt-top,l-left:r-left]|=mask[t-b:bt-b,l-a:r-a]
    if not found:return []
    blocked=cv2.dilate(blocked,np.ones((25,25),np.uint8))
    crop=cleaned[top:bottom,left:right]
    # Only move into plain light background; no guessed relocation over art.
    safe=((crop.min(axis=2)>215)&(np.ptp(crop,axis=2)<32)&(blocked==0)).astype(np.uint8)*255
    for other in others:
        ox,oy,ow,oh=(int(other[k]) for k in ('x','y','width','height'))
        l,t,r,bt=max(left,ox-6),max(top,oy-6),min(right,ox+ow+6),min(bottom,oy+oh+6)
        if r>l and bt>t:safe[t-top:bt-top,l-left:r-left]=0
    candidates=[]
    offsets=range(-radius,radius+1,8)
    for dx,dy in sorted([(dx,dy) for dx in offsets for dy in offsets]+[(0,0)],key=lambda v:v[0]**2+v[1]**2):
        a,b=x+dx,y+dy;c,d=a+w,b+h
        if a<left or b<top or c>right or d>bottom:continue
        mask=safe[b-top:d-top,a-left:c-left].copy()
        if np.count_nonzero(mask)<w*h*.85:continue
        candidates.append(((a,b,c,d),mask))
        if len(candidates)>=24:break
    return candidates
