"""Enlarge source replacement logos without touching nearby dialogue/borders."""
import cv2
import numpy as np
from PIL import Image
from branding_v51 import overlaps
from branding_ink_guard import intersects_protected


def fit_larger_logo(logo,bounds,source_mask,repaired,protected):
    h,w=repaired.shape[:2];x1,y1,x2,y2=bounds
    # Slightly stronger Ryukomik presence while retaining every existing
    # collision/border check below.
    desired=min(round(w*.29),round((x2-x1)*.96))
    # A modest local expansion, never a relocation to a random page corner.
    pad=max(8,round(w*.12))
    cx=(x1+x2)/2;cy=(y1+y2)/2
    roi=(max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad))
    rx,ry,rr,rb=roi
    gray=cv2.cvtColor(repaired[ry:rb,rx:rr],cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,100,180)
    # Long straight lines are likely panel borders. Protect their continuation.
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=35,minLineLength=max(40,round(w*.14)),maxLineGap=5)
    borders=np.zeros(edges.shape,np.uint8)
    if lines is not None:
        for line in lines[:,0]:
            ax,ay,bx,by=map(int,line)
            probe=np.zeros(edges.shape,np.uint8);cv2.line(probe,(ax,ay),(bx,by),255,3)
            # Bright texture strokes aren't panel borders. Require a sustained
            # dark line, rather than every edge in a detailed illustration.
            if float(np.mean(gray[probe>0]<100))<.20:continue
            cv2.line(borders,(ax,ay),(bx,by),255,3)
    for width in range(desired,39,-2):
        mark=logo.resize((width,max(1,round(logo.height*width/logo.width))),Image.Resampling.LANCZOS)
        for offset in (0,-round(w*.025),round(w*.025),-round(w*.045),round(w*.045),-round(w*.07),round(w*.07)):
            px=round(cx-mark.width/2);py=round(cy-mark.height/2)+offset
            rect=(px,py,px+mark.width,py+mark.height)
            if px<rx or py<ry or rect[2]>rr or rect[3]>rb:continue
            ink=np.array(mark.getchannel('A'))>0
            original=source_mask[py:rect[3],px:rect[2]]>0
            blocked=False
            for p in protected:
                if not overlaps(rect,p['bbox'],5):continue
                if intersects_protected(ink&~original,(px,py),p,5):blocked=True;break
            if blocked:continue
            border=borders[py-ry:rect[3]-ry,px-rx:rect[2]-rx]>0
            if np.any(ink&~original&border):continue
            return mark,(px,py),{'requested_logo_width':desired,'logo_width':width,'placement_engine':'v5.3','local_expansion_limit':pad}
    raise RuntimeError('V5.3 cannot place readable enlarged logo safely')
