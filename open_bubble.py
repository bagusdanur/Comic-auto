"""Recover oval balloon boundaries only when a substantial visible arc supports them."""
import cv2
import numpy as np

def open_bubble_mask(reference,box,text_boxes):
    x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
    ph,pw=reference.shape[:2]
    a,b,c,d=max(0,x-200),max(0,y-240),min(pw,x+w+200),min(ph,y+h+240)
    crop=reference[b:d,a:c]
    dark=(crop.max(axis=2)<140).astype(np.uint8)*255
    for tb in [box,*text_boxes]:
        tx,ty,tw,th=(int(tb[k]) for k in ('x','y','width','height'))
        l,t,r,bt=max(a,tx-4),max(b,ty-4),min(c,tx+tw+4),min(d,ty+th+4)
        if r>l and bt>t:dark[t-b:bt-b,l-a:r-a]=0
    contours,_=cv2.findContours(dark,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    proposals=[]
    for contour in sorted(contours,key=len,reverse=True)[:12]:
        if len(contour)<100:continue
        proposals.append(contour)
        for length in (160,280,440,640):
            for start in range(0,len(contour)-length+1,80):
                proposals.append(contour[start:start+length])
    distance=cv2.distanceTransform((dark==0).astype(np.uint8),cv2.DIST_L2,3)
    accepted=[]
    for contour in proposals:
        if len(contour)<100:continue
        ellipse=cv2.fitEllipse(contour)
        (cx,cy),(ew,eh),angle=ellipse
        if min(ew,eh)<max(60,h) or max(ew,eh)>min(600,pw*.8):continue
        if cx-ew/2<0 or cy-eh/2<0 or cx+ew/2>=c-a or cy+eh/2>=d-b:continue
        points=contour[:,0,:].astype(float)-[cx,cy]
        theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        uv=points@rot/np.array([ew/2,eh/2])
        radii=np.linalg.norm(uv,axis=1)
        good=np.abs(radii-1)<.08
        if good.mean()<.85:continue
        bins=np.unique(((np.arctan2(uv[good,1],uv[good,0])+np.pi)*24/(2*np.pi)).astype(int)%24)
        if len(bins)<7:continue
        angles=np.linspace(0,2*np.pi,96,endpoint=False)
        ring=np.column_stack((np.cos(angles)*ew/2,np.sin(angles)*eh/2))@rot.T+[cx,cy]
        ix=np.rint(ring[:,0]).astype(int);iy=np.rint(ring[:,1]).astype(int)
        if ix.min()<0 or iy.min()<0 or ix.max()>=c-a or iy.max()>=d-b:continue
        support=float((distance[iy,ix]<4).mean())
        if support<.65:continue
        mask=np.zeros(dark.shape,np.uint8);cv2.ellipse(mask,ellipse,255,-1)
        if np.count_nonzero(mask[y-b:y+h-b,x-a:x+w-a])<w*h*.8:continue
        inner=cv2.erode(mask,np.ones((11,11),np.uint8))>0
        if not inner.any() or (crop[inner].min(axis=1)>205).mean()<.75:continue
        accepted.append((support,mask))
    if not accepted:return None
    return (a,b,c,d),max(accepted,key=lambda candidate:candidate[0])[1]
