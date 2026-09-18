"""Estimate dominant RAW glyph fill, excluding its reconstructed background.

This is deliberately separate from bubble/background brightness. Color samples
must have been removed by cleaning and form repeated glyph-sized components.
"""
import cv2
import numpy as np


def source_text_color(reference,cleaned,box):
    x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
    x1,y1=max(0,x),max(0,y);x2,y2=min(reference.shape[1],x+w),min(reference.shape[0],y+h)
    raw=reference[y1:y2,x1:x2];back=cleaned[y1:y2,x1:x2]
    fallback=(255,255,255) if back.size and float(np.median(back))<100 else (0,0,0)
    result={'color':fallback,'color_source':'BACKGROUND_CONTRAST_FALLBACK','color_confidence':0.0,'fill_mode':'solid'}
    if raw.size==0 or raw.shape!=back.shape:return result
    delta=np.max(np.abs(raw.astype(np.int16)-back.astype(np.int16)),axis=2)
    removed=delta>45
    hsv=cv2.cvtColor(raw,cv2.COLOR_RGB2HSV)
    background_hsv=cv2.cvtColor(back,cv2.COLOR_RGB2HSV)
    hue_delta=np.abs(hsv[:,:,0].astype(int)-background_hsv[:,:,0].astype(int))
    hue_delta=np.minimum(hue_delta,180-hue_delta)
    # Antialiased black letters on red art have reddish edge pixels. They are
    # blends of the background, not red glyph fill.
    distinct_color=(background_hsv[:,:,1]<70)|(hue_delta>9)|(hsv[:,:,1].astype(int)>background_hsv[:,:,1].astype(int)+30)
    chromatic=removed&distinct_color&(hsv[:,:,1]>95)&(hsv[:,:,2]>65)
    # Keep translated lettering on the established solid-color treatment.
    # Gradient reconstruction was visually inconsistent between panels, so a
    # colored RAW glyph is reduced to one representative fill below.
    # Quantized hue is stable under antialiasing. Red wraps around HSV hue 0.
    hue=((hsv[:,:,0].astype(int)+5)//10)%18
    histogram=np.bincount(hue[chromatic],minlength=18)
    best=int(histogram.argmax())
    color_mask=chromatic&(hue==best)
    n,labels,stats,_=cv2.connectedComponentsWithStats(color_mask.astype(np.uint8))
    good=[]
    for label in range(1,n):
        bx,by,bw,bh,area=stats[label]
        if area>=3 and area<raw.shape[0]*raw.shape[1]*.25 and bh>=3:
            good.append(label)
    pixels=np.isin(labels,good)
    count=int(np.count_nonzero(pixels))
    # Repeated components reject a lone patch of colored art in the OCR box.
    if len(good)>=2 and count>=max(12,int(np.count_nonzero(removed)*.08)):
        samples=raw[pixels]
        fill=tuple(int(v) for v in np.median(samples,axis=0))
        return {'color':fill,'color_source':'RAW_GLYPH_COLOR','color_confidence':round(min(.99,count/max(1,np.count_nonzero(chromatic))),3),'fill_mode':'solid'}
    # RAW achromatic cores: preserve white lettering on light artwork with an
    # outline as well as black text on dark colored backgrounds.
    neutral=removed&(hsv[:,:,1]<70)
    dark=neutral&(hsv[:,:,2]<85);light=neutral&(hsv[:,:,2]>210)
    nd,nl=int(dark.sum()),int(light.sum())
    if max(nd,nl)>=max(12,int(removed.sum()*.15)):
        dominance=max(nd,nl)/max(1,nd+nl)
        # When black cores and white outline rings are nearly tied, counting
        # pixels cannot identify which one is the fill. Prefer the established
        # background-contrast colour; this preserves black lettering with a
        # white keyline instead of accidentally inverting it to white fill.
        if nd and nl and dominance < .58:
            fill=fallback
            source='RAW_GLYPH_NEUTRAL_CORE_TIEBREAK'
        else:
            fill=(255,255,255) if nl>nd else (0,0,0)
            source='RAW_GLYPH_NEUTRAL'
        return {'color':fill,'color_source':source,'color_confidence':round(dominance,3),'fill_mode':'solid'}
    return result
