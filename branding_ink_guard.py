"""Local, measured translated-ink protection; layout boxes are not glyphs."""
import cv2
import numpy as np


def measured_ink_region(item,rendered,cleaned):
    result=dict(item)
    if cleaned is None or rendered.shape!=cleaned.shape:return result
    b=item['bbox'];x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
    x1,y1=max(0,x),max(0,y);x2,y2=min(rendered.shape[1],x+w),min(rendered.shape[0],y+h)
    if x2<=x1 or y2<=y1:return result
    ink=np.any(np.abs(rendered[y1:y2,x1:x2].astype(np.int16)-cleaned[y1:y2,x1:x2].astype(np.int16))>3,axis=2)
    if not np.any(ink):return result  # missing evidence must not disable safety
    result['bbox']={'x':x1,'y':y1,'width':x2-x1,'height':y2-y1}
    result['_ink_mask']=cv2.dilate(ink.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
    return result


def intersects_protected(candidate,origin,region,padding=2):
    """candidate is a local boolean/uint8 mask at page-space origin (x,y)."""
    ox,oy=origin;b=region['bbox'];x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
    ink=region.get('_ink_mask')
    pad=0 if ink is not None else padding
    left,top=max(ox,x-pad),max(oy,y-pad)
    right,bottom=min(ox+candidate.shape[1],x+w+pad),min(oy+candidate.shape[0],y+h+pad)
    if right<=left or bottom<=top:return False
    overlap=candidate[top-oy:bottom-oy,left-ox:right-ox]>0
    if ink is not None:overlap=overlap&ink[top-y:bottom-y,left-x:right-x]
    return bool(np.any(overlap))
