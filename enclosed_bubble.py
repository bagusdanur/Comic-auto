"""Find a genuinely enclosed light balloon, including colored spiky borders."""
import cv2
import numpy as np

def enclosed_bubble_mask(reference,box,others):
    x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
    ph,pw=reference.shape[:2]
    px=max(100,round(w*.65));py=max(110,round(h*1.3))
    a,b,c,d=max(0,x-px),max(0,y-py),min(pw,x+w+px),min(ph,y+h+py)
    crop=reference[b:d,a:c]
    light=((crop.min(axis=2)>215)&(np.ptp(crop,axis=2)<32)).astype(np.uint8)
    # Close small breaks in the border, never paint a bounding rectangle white.
    light=cv2.morphologyEx(light,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    n,labels,stats,_=cv2.connectedComponentsWithStats(light)
    best=None
    for label in range(1,n):
        sx,sy,sw,sh,area=stats[label]
        if sx==0 or sy==0 or sx+sw>=c-a or sy+sh>=d-b:continue
        overlap=np.count_nonzero(labels[y-b:y+h-b,x-a:x+w-a]==label)
        if overlap< w*h*.35 or area<w*h*.5:continue
        if best is None or overlap>best[0]:best=(overlap,label)
    if best is None:return None
    mask=(labels==best[1]).astype(np.uint8)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask,contours,-1,255,cv2.FILLED)
    for other in others:
        ox,oy,ow,oh=(int(other[k]) for k in ('x','y','width','height'))
        l,t,r,bt=max(0,ox-a-5),max(0,oy-b-5),min(c-a,ox+ow-a+5),min(d-b,oy+oh-b+5)
        if r>l and bt>t:mask[t:bt,l:r]=0
    mask=cv2.erode(mask,np.ones((9,9),np.uint8))
    yy,xx=np.where(mask)
    if not len(xx):return None
    l,t,r,bt=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
    return (a+l,b+t,a+r,b+bt),mask[t:bt,l:r]
