"""Inpaint Qimanga banners and composite a larger transparent brand logo."""
import re
import os
import numpy as np
from PIL import Image

def is_qimanga_group(group,source=''):
    tokens=[re.sub(r'\s+','',r.get('text','')).strip('.,!|').upper() for r in group]
    return any(re.fullmatch(r'[QCG]IMAN[GE]A\.CO[MN]|QIMEERA\.COM',t) for t in tokens) or ('onimusha' in str(source).lower() and any(t in ('MA.COM','OUANCA.COM') for t in tokens))

def cover_qimanga(image,group,logo,top,bottom,inpaint=None):
    boxes=[r['bbox'] for r in group]
    right=max(b['x']+b['width'] for b in boxes);y=min(b['y'] for b in boxes)
    low=max(b['y']+b['height'] for b in boxes);height=low-y
    text_width=max(b['width'] for b in boxes)
    # Include the left emblem and complete translucent banner even when OCR
    # only sees a suffix. This rectangle is an inpainting mask, not a backing.
    left=max(0,round(right-max(text_width,height*4.8)-height*1.6))
    rect=(left,max(top,round(y-height*.45)),min(image.width,round(right+height*.35)),min(bottom,round(low+height*.3)))
    a,b,c,d=rect
    # Repair a local context crop from the unbranded render; no opaque backing.
    pad=128
    l,t,r,bt=max(0,a-pad),max(top,b-pad),min(image.width,c+pad),min(bottom,d+pad)
    original=np.array(image.convert('RGB'))
    context=original[t:bt,l:r].copy();mask=np.zeros(context.shape[:2],np.uint8)
    mask[b-t:d-t,a-l:c-l]=255
    if inpaint is None:
        from xianscan_client import XianScanClient
        inpaint=XianScanClient(os.getenv('XIANSCAN_URL','http://127.0.0.1:8123')).inpaint_mask
    repaired=inpaint(context,mask)
    if repaired.shape!=context.shape or np.any(np.any(repaired!=context,axis=2)&(mask==0)):
        raise RuntimeError('Qimanga inpainting changed pixels outside its mask')
    original[b:d,a:c]=repaired[b-t:d-t,a-l:c-l]
    canvas=Image.fromarray(original).convert('RGBA')
    scale=min((c-a)*.9/logo.width,(bottom-top)/logo.height)
    mark=logo.resize((max(1,round(logo.width*scale)),max(1,round(logo.height*scale))),Image.Resampling.LANCZOS)
    pos=(a+(c-a-mark.width)//2,max(top,min(bottom-mark.height,b+(d-b-mark.height)//2)))
    canvas.alpha_composite(mark,pos)
    return canvas.convert('RGB'),rect
