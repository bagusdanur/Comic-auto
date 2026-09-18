"""Conservative source-brand cleanup and gutter-only replacement branding.

Uncertain source marks remain intact. A separate footer is preferable to
guessing where artwork or an empty speech balloon may safely be overwritten.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

VERSION = 'v5.1'
ANCHOR = re.compile(r'(?:https?://\S+|www\.\S+|\b[a-z0-9][a-z0-9-]*\.(?:com|org|net|gg|to|my\.id)\b|\b(?:[a-z]{2,}\s*scans|thunder\s*scans?|asura\s*(?:scans|comics)|reaper\s*scans|flame\s*comics|manhwafreak|qiscans|madara\s*scans?|eva\s*scans?)\b)', re.I)


def confirmed_credit(text):
    # Ordinary dialogue mentioning scans, reading, support or a website is not
    # evidence of a publisher mark. Long sentences containing a URL stay safe.
    # Qimanga's translucent lettering is repeatedly read as C/Q and G/E,
    # with the final M read as N. Only accept the complete domain token.
    compact=re.sub(r'\s+','',text).strip('.,!|').upper()
    qimanga=bool(re.fullmatch(r'[QC]IMAN[GE]A\.CO[MN]',compact))
    vortex=bool(re.fullmatch(r'V[OU]RTEXSCANS\.C[OU]M',compact))
    # EvaScans used .org on You Like Someone With That Face. OCR commonly
    # merges EVA SCANS or reads O/G as 0/C in the small source watermark.
    evascans=bool(re.fullmatch(r'E[VY]ASCAN[S5](?:\.?[O0]R[GC])?',compact))
    madarascans=bool(re.fullmatch(r'MADARA(?:SCAN[S5])?(?:\.?C[O0][MN])?',compact))
    if re.search(r'V[OU]RTEXSCANS\.C[OU]M',compact) and not vortex and not re.search(r'READ|SUPPORT|CHAPTER|WEBSITE|EXCLUSIVELY',compact):
        return False
    return qimanga or vortex or evascans or madarascans or (bool(ANCHOR.search(text)) and len(text.split()) <= 14)


def overlaps(rect, box, pad=0):
    x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
    return rect[0] < x+w+pad and rect[2] > x-pad and rect[1] < y+h+pad and rect[3] > y-pad


def clean_credit(image, region, protected):
    """Only reconstruct isolated marks surrounded by a uniform, light rim.

    No fill of an expanded union of OCR boxes; only detected foreground pixels
    within the verified local context may change. Busy/ambiguous areas abstain.
    """
    w,h=image.size
    b=region['bbox'];x,y,bw,bh=(int(b[k]) for k in ('x','y','width','height'))
    decision={'source_bbox':b,'text':region.get('text',''),'action':'PRESERVED'}
    if bw<=0 or bh<=0 or bw>w*.65 or bh>w*.22:
        return {**decision,'reason':'LARGE_OR_INVALID_SOURCE_MARK'}
    pad=max(10,round(w*.035))
    rect=(max(0,x-pad),max(0,y-pad),min(w,x+bw+pad),min(h,y+bh+pad))
    if rect[0]==0 or rect[1]==0 or rect[2]==w or rect[3]==h:
        return {**decision,'reason':'SOURCE_BOUNDARY_NOT_VISIBLE'}
    if any(overlaps(rect,p['bbox'],6) for p in protected):
        return {**decision,'reason':'NEAR_PROTECTED_TEXT_OR_SURFACE'}
    crop=np.array(image.convert('RGB').crop(rect))
    rim=np.concatenate((crop[:4].reshape(-1,3),crop[-4:].reshape(-1,3),crop[:,:4].reshape(-1,3),crop[:,-4:].reshape(-1,3)))
    background=np.median(rim,axis=0)
    deviation=np.max(np.abs(rim.astype(float)-background),axis=1)
    if background.min()<225 or np.quantile(deviation,.99)>10:
        return {**decision,'reason':'TEXTURED_OR_UNCERTAIN_BACKGROUND'}
    foreground=(np.max(np.abs(crop.astype(float)-background),axis=2)>22).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(foreground)
    seed=labels[y-rect[1]:y-rect[1]+bh,x-rect[0]:x-rect[0]+bw]
    selected=set(np.unique(seed))-{0}
    mask=np.zeros(foreground.shape,np.uint8)
    for label in selected:
        lx,ly,lw,lh,area=stats[label]
        if lx<3 or ly<3 or lx+lw>crop.shape[1]-3 or ly+lh>crop.shape[0]-3:
            return {**decision,'reason':'FOREGROUND_CONTINUES_OUTSIDE_SOURCE'}
        mask[labels==label]=255
    if not np.any(mask):
        return {**decision,'reason':'NO_VISIBLE_MARK'}
    mask=cv2.dilate(mask,np.ones((3,3),np.uint8))
    restored=crop.copy()
    restored[mask>0]=np.round(background).astype(np.uint8)
    image.paste(Image.fromarray(restored).convert('RGBA'),(rect[0],rect[1]))
    return {**decision,'action':'CLEANED','reason':'ISOLATED_MARK_ON_VERIFIED_FLAT_BACKGROUND','mask_pixels':int(np.count_nonzero(mask)),'context':list(rect),'outside_mask_changed_pixels':0}


def gutter_position(image, mark, protected):
    """Require a full-width blank band, not merely a white rectangle/bubble."""
    w,h=image.size
    pad=max(12,round(w*.025))
    band_height=mark.height+2*pad
    if band_height>h:return None
    # Row summaries keep the search O(page pixels), not O(candidates * pixels).
    rgb=np.asarray(image.convert('RGB'))
    blank=(rgb.min(axis=2)>=246).mean(axis=1)>=.998
    bad=np.concatenate(([0],np.cumsum(~blank)))
    candidates=[]
    for top in range(0,h-band_height+1,max(4,pad//2)):
        bottom=top+band_height
        if bad[bottom]-bad[top]:continue
        if any(overlaps((0,top,w,bottom),p['bbox'],pad) for p in protected):continue
        # Stable placement: prefer the lowest safe gutter, aligned to the right.
        candidates.append((w-pad-mark.width,top+pad))
    return candidates[-1] if candidates else None


def apply_branding_v51(image_path,ocr_path,logo_path,text_path,*,first_page=False,banner_path=None):
    with Image.open(image_path) as opened:image=opened.convert('RGBA')
    source_size=list(image.size)
    ocr=json.loads(Path(ocr_path).read_text(encoding='utf-8'))
    regions=ocr.get('regions',[])
    credits=[r for r in regions if confirmed_credit(str(r.get('text','')))]
    protected=[r for r in regions if r not in credits]
    translations_path=Path(ocr_path).with_name('translations.json')
    if translations_path.exists():
        for r in json.loads(translations_path.read_text(encoding='utf-8')).get('groups',[]):
            if r.get('bbox') and not confirmed_credit(str(r.get('text',''))):protected.append(r)
    # Expanded render boxes protect actual translated text, not only RAW boxes.
    for folder in (Path(image_path).parent,Path(ocr_path).parent):
        report_path=folder/'typeset-report.json'
        if report_path.exists():
            for r in json.loads(report_path.read_text(encoding='utf-8')).get('results',[]):
                if r.get('bbox') and r.get('status')=='TYPESET':protected.append(r)
    # Avoid empty portions of recognized balloons as well as their text.
    surface_path=Path(ocr_path).with_name('surface-map.json')
    if surface_path.exists():
        for s in json.loads(surface_path.read_text(encoding='utf-8')).get('surfaces',[]):
            if s.get('source_credit') or s.get('surface_type')!='BUBBLE_FLAT':continue
            polygon=s.get('polygon') or s.get('safe_polygon')
            if polygon:
                pts=np.asarray(polygon);lo=pts.min(axis=0);hi=pts.max(axis=0)
                protected.append({'bbox':dict(zip(('x','y','width','height'),map(int,(*lo,*(hi-lo)))))})
    report={'branding_engine':VERSION,'input_dimensions':source_size,'decisions':[],
            'replacements':[],'source_header_trimmed_at':None,'top_credit_cleared':None,
            'source_footer_trimmed_at':None,'text_watermark':None,'banner':None,
            'detected_credit_texts':[r.get('text','') for r in credits]}
    for region in credits:
        decision=clean_credit(image,region,protected)
        report['decisions'].append(decision)
    # Keep every original panel: do not estimate header/footer crop boundaries.
    # At most one modest logo per page. The last-page banner supplies its brand.
    if (credits or first_page) and banner_path is None:
        with Image.open(logo_path) as opened:logo=opened.convert('RGBA')
        content=logo.getchannel('A').getbbox()
        if not content:raise ValueError('Brand logo is fully transparent')
        logo=logo.crop(content)
        target=min(max(1,round(image.width*.18)),image.width-2*max(12,round(image.width*.025)))
        if target<1:raise ValueError('Page too narrow for branding')
        logo=logo.resize((target,max(1,round(logo.height*target/logo.width))),Image.Resampling.LANCZOS)
        position=gutter_position(image,logo,protected+credits)
        mode='VERIFIED_FULL_WIDTH_GUTTER'
        if position is None:
            pad=max(12,round(image.width*.025))
            expanded=Image.new('RGBA',(image.width,image.height+logo.height+2*pad),'white')
            expanded.alpha_composite(image)
            position=(image.width-pad-logo.width,image.height+pad)
            image=expanded;mode='ADDED_BRANDING_STRIP'
        image.alpha_composite(logo,position)
        report['replacements'].append({'logo':[*position,logo.width,logo.height],'mode':mode})
    if banner_path is not None:
        with Image.open(banner_path) as opened:banner=opened.convert('RGBA')
        banner=banner.resize((image.width,max(1,round(banner.height*image.width/banner.width))),Image.Resampling.LANCZOS)
        expanded=Image.new('RGBA',(image.width,image.height+banner.height),'white')
        expanded.alpha_composite(image);expanded.alpha_composite(banner,(0,image.height))
        report['banner']=[0,image.height,banner.width,banner.height];image=expanded
    report['output_dimensions']=list(image.size)
    image.convert('RGB').save(image_path,format='PNG',optimize=True)
    return report
