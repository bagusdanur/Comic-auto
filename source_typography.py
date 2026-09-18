"""Conservative RAW typography cues for V5 comic typesetting."""
from __future__ import annotations

import cv2
import numpy as np


def source_typography(reference, cleaned, box, semantic_style="dialogue"):
    x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
    x1,y1=max(0,x),max(0,y);x2,y2=min(reference.shape[1],x+w),min(reference.shape[0],y+h)
    raw=reference[y1:y2,x1:x2];back=cleaned[y1:y2,x1:x2]
    semantic_style=semantic_style if semantic_style in {'dialogue','narration','thought','shout','sfx','chat'} else 'dialogue'
    result={'semantic_style':semantic_style,'font_variant':'regular','style_source':'semantic_fallback','style_confidence':.60,
            'raw_bold_score':0.0,'raw_italic_score':0.0}
    if semantic_style=='thought':result.update(font_variant='italic',style_confidence=.76)
    elif semantic_style=='shout':result.update(font_variant='bold_italic',style_confidence=.72)
    elif semantic_style in {'sfx','chat'}:result.update(font_variant='bold',style_confidence=.76)
    if raw.size==0 or raw.shape!=back.shape:return result
    # Estimate glyphs against the reconstructed local background. Inpainting
    # changes a halo around letters, so raw-vs-clean delta by itself is too
    # broad for slant/weight measurement.
    background=np.median(back.reshape(-1,3),axis=0)
    contrast=np.max(np.abs(raw.astype(np.int16)-background.astype(np.int16)),axis=2)>52
    gray=np.mean(raw,axis=2);background_gray=float(np.mean(background))
    if background_gray>=170:contrast&=gray<background_gray-38
    elif background_gray<=90:contrast&=gray>background_gray+38
    n,labels,stats,_=cv2.connectedComponentsWithStats(contrast.astype(np.uint8))
    keep=[]
    for label in range(1,n):
        _,_,cw,ch,area=stats[label]
        if 3<=area<=raw.shape[0]*raw.shape[1]*.22 and ch>=3:keep.append(label)
    ink=np.isin(labels,keep).astype(np.uint8)
    if int(ink.sum())<30:return result
    distance=cv2.distanceTransform(ink,cv2.DIST_L2,3)
    radii=distance[distance>0]
    bold_score=float(np.percentile(radii,80))
    edges=cv2.Canny(ink*255,50,150)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(5,h//10),minLineLength=max(6,h//5),maxLineGap=3)
    slants=[]
    if lines is not None:
        for x0,y0,x3,y3 in lines[:,0]:
            dy=y3-y0;dx=x3-x0
            if abs(dy)>=max(6,h*.16) and abs(dx)<=abs(dy)*.75:
                slants.append(dx/dy)
    italic_score=abs(float(np.median(slants))) if len(slants)>=2 else 0.0
    result['raw_bold_score']=round(bold_score,3);result['raw_italic_score']=round(italic_score,3)
    # Require a pronounced and repeated slant. The earlier 0.16 threshold
    # mistook jagged uppercase strokes for italics, especially in shout text.
    raw_italic=len(slants)>=4 and italic_score>=.22
    raw_bold=bold_score>=2.25
    # Once a usable RAW glyph sample exists it is authoritative, including a
    # positive regular result. This prevents semantic "shout" guesses from
    # making ordinary uppercase dialogue bold-italic.
    variant='bold_italic' if raw_italic and raw_bold else 'italic' if raw_italic else 'bold' if raw_bold else 'regular'
    confidence=min(.95,.68+(len(slants)>=3)*.08+abs(bold_score-2.25)*.05)
    if semantic_style in {'sfx','chat'} and variant=='regular':variant='bold'
    result.update(font_variant=variant,style_source='raw_visual',style_confidence=round(confidence,3))
    return result


def select_font(args, typography):
    # Preserve detected RAW typography in reports, but render every region
    # with one regular face so merged/overlapping bubbles never mix fonts.
    del typography
    return args.font_dialogue or args.font
