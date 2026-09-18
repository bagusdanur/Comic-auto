"""V5.2: structural promo-card cropping, source text cleanup and pill replacement."""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from branding_v51 import confirmed_credit, overlaps
from branding_ink_guard import measured_ink_region, intersects_protected
from qimanga_cover import is_qimanga_group,cover_qimanga

VERSION='v5.2'
PROMO=re.compile(r'(?:read\s*(?:at|on|only|this)|you\s*can\s*read|support|website|fastest\s+releases|check\s+other\s+series|thunder|scans|discord|quality|translation|publish|warning)',re.I)


def promo_text(text):
    """Tolerate the common Y/C OCR confusion in tiny Thunder captions."""
    value=str(text)
    return bool(PROMO.search(value)
                or re.search(r'\b[YC]?OU\s*CAN\s*READ\s*(?:THE\s+)?CHAPTER',value,re.I)
                or re.fullmatch(r'\s*(?:CH)?APTER\s+ON[:.]?\s*',value,re.I))


def standalone_promo(reference, regions):
    """Recognize a short, independently uploaded end-credit advertisement."""
    h,w=reference.shape[:2]
    texts=[str(r.get('text','')).upper() for r in regions]
    return (h<=w*1.4
            and any('READ ONLY AT' in str(r.get('text','')).upper() and r['bbox']['y']<h*.15 for r in regions)
            and any('DISCORD' in t for t in texts)
            and any('THUNDERSCANS.COM' in t for t in texts)
            and any('SUPPORT' in t for t in texts)
            and float((reference.mean(axis=2)<90).mean())>.55
            and not any(len(t)>15 and not re.search(r'READ|SUPPORT|DISCORD|THUNDER|SCANS|REACH OUT|OUR SITE',t) for t in texts))


def standalone_madara_promo(reference, regions):
    """Recognize MadaraScans' short full-page chapter advertisement."""
    h,w=reference.shape[:2]
    joined=' '.join(str(r.get('text','')).upper() for r in regions)
    compact=re.sub(r'\s+','',joined)
    return (h <= w*.75
            and 'YOU CAN READ' in joined
            and 'CHAPTERS' in joined
            and 'FROM' in joined
            and bool(re.search(r'MADARA(?:SCAN[S5])?\.?C[O0][MN]',compact))
            and float((reference.mean(axis=2)<85).mean())>.45)


def standalone_eva_promo(reference, regions):
    """Recognize EvaScans' full illustrated end-credit card."""
    h,w=reference.shape[:2]
    joined=' '.join(str(r.get('text','')).upper() for r in regions)
    compact=re.sub(r'\s+','',joined)
    return (w*.9 <= h <= w*1.8
            and bool(re.search(r'E[VY]ASCAN[S5]',compact))
            and 'READ' in joined and ('ONLY AT' in joined or 'READ ONLY' in joined)
            and 'SUPPORT' in joined
            and ('REACH OUT' in joined or 'DISCORD' in joined)
            and float((reference[:,:,0]>reference[:,:,1]*1.35).mean())>.35)


def stacked_source_promo_header(reference, regions):
    """Return the safe lower edge of a Madara+Eva stacked intro, if present."""
    h,w=reference.shape[:2]
    head=[r for r in regions if int(r.get('bbox',{}).get('y',h)) < min(h*.48,w*4.5)]
    joined=' '.join(str(r.get('text','')).upper() for r in head)
    compact=re.sub(r'\s+','',joined)
    has_madara=bool(re.search(r'MADARA(?:SCAN[S5])?(?:\.?C[O0][MN])?',compact))
    has_eva=bool(re.search(r'E[VY]ASCAN[S5](?:\.?[O0]R[GC])?',compact))
    eva_marks=[r for r in head if re.search(r'E\s*[VY]\s*A\s*SCAN[S5]?|EVASCANS\s*\.\s*[O0]R[GC]',str(r.get('text','')),re.I)]
    early_text=' '.join(str(r.get('text','')).upper() for r in head if int(r['bbox']['y'])<w*2)
    eva_only_intro=(len(eva_marks)>=2
                    and ('YOU CAN READ' in early_text or min(int(r['bbox']['y']) for r in eva_marks)<w*.8)
                    and ('WEBSITE' in early_text or 'ONLY AT OUR' in early_text)
                    and ('DISCORD' in early_text or 'PAYPAL' in early_text))
    if eva_only_intro:
        promo=[r for r in head if int(r['bbox']['y'])<w*2]
        last=max(int(r['bbox']['y'])+int(r['bbox']['height']) for r in promo)
        white=(reference.min(axis=2)>242).mean(axis=1)
        for edge in range(min(h-20,last+6),min(h-20,last+round(w*1.5))):
            if np.all(white[edge:edge+20]>.985):return edge
    anchors=sum(bool(re.search(pattern,joined)) for pattern in (
        r'HIGH QUALITY',r'JOIN OUR',r'READ ONLY ON OUR SITE',
        r'YOU CAN READ CHAPTERS',r'ONLY AT OUR WEBSITE'))
    # OCR often reverses tiny icon captions ("OUR JOIN", "ONLY ... READ").
    # Two independent promo phrases plus both source brands are sufficient;
    # ordinary story dialogue cannot satisfy this combination accidentally.
    if not (has_madara and has_eva and anchors>=2):return 0
    promo=[r for r in head if int(r['bbox']['y']) < w*4.15]
    if not promo:return 0
    last=max(int(r['bbox']['y'])+int(r['bbox']['height']) for r in promo)
    white=(reference.min(axis=2)>242).mean(axis=1)
    for edge in range(min(h-20,last+6),min(h-20,last+round(w*.8))):
        if np.all(white[edge:edge+20]>.985):return edge
    return 0


def short_vortex_end_promo(reference, regions):
    """Recognize a compact page that is only a Vortex end-credit ad."""
    h,w=reference.shape[:2]
    joined=' '.join(str(r.get('text','')).upper() for r in regions)
    if h>w*.9:return False
    if 'DEAR READERS' not in joined or 'VORTEXSCANS.ORG' not in joined:return False
    if not re.search(r'UNLOCK UP TO|LATEST CHAPTER|EXCLUSIVELY ON',joined):return False
    allowed=re.compile(r'DEAR READERS|DIVE DEEPER|STORY|UNLOCK|LATEST|CHAPTER|HAPPY|READING|EXCLUSIVELY|VORTEXSCANS|ORG|AND|THE|INTO|UP|TO|ON|[-,!.]')
    unknown=[t for t in (str(r.get('text','')).upper() for r in regions) if len(t)>8 and allowed.sub('',t).strip()]
    dark=float((reference.mean(axis=2)<80).mean())
    red=float(((reference[:,:,0]>130)&(reference[:,:,1]<90)&(reference[:,:,2]<90)).mean())
    return not unknown and dark>.25 and red>.03


def card_bounds(reference,regions):
    """Require both explicit advertising anchors and a full-width visual edge."""
    h,w=reference.shape[:2];top=stacked_source_promo_header(reference,regions);bottom=h
    gray=reference.mean(axis=2)
    dark=(gray[:,round(w*.05):round(w*.95)]<70).mean(axis=1)
    text=lambda r:str(r.get('text',''))
    warnings=[r for r in regions if 'WARNING' in text(r).upper() and r['bbox']['y']<min(h*.2,w)]
    domains=[r for r in regions if confirmed_credit(text(r))]
    eva=lambda r:bool(re.search(r'E\s*[VY]\s*A\s*SCAN[S5]?|EVASCANS\s*\.\s*[O0]R[GC]',text(r),re.I))
    eva_header=[r for r in regions if r['bbox']['y']<w*1.0]
    eva_header_text=' '.join(text(r) for r in eva_header).upper()
    if (any(eva(r) for r in eva_header)
        and re.search(r'READ|WEBSITE',eva_header_text)
        and re.search(r'DISCORD|PAYPAL',eva_header_text)):
        last=max(r['bbox']['y']+r['bbox']['height'] for r in eva_header)
        white=(reference.min(axis=2)>242).mean(axis=1)
        for edge in range(int(last)+8,min(h-16,int(last+w*.9))):
            if np.all(white[edge:edge+16]>.985):
                top=max(top,edge);break
    # Oversized Vortex advertisements surround (but are not) the story's
    # "To be continued" sign-off. Trim only independently anchored ad bands.
    vortex=lambda r:bool(re.search(r'VORTEX\s*SCANS\s*\.\s*(?:COM|ORG)',text(r),re.I))
    starts_vortex=[r for r in regions if 'READ THIS SERIES FIRST AT' in text(r).upper() and r['bbox']['y']<w*.25]
    if starts_vortex:
        ads=[r for r in regions if vortex(r) and r['bbox']['y']<w*.5]
        if ads:
            end=max(r['bbox']['y']+r['bbox']['height'] for r in starts_vortex+ads)
            unknown=[r for r in regions if r['bbox']['y']<end and r not in starts_vortex+ads]
            # Flat full-width background below the ad, with no story content.
            for edge in range(int(end)+12,min(h-8,int(end)+48)) if not unknown else []:
                strip=reference[edge:edge+8]
                if float(np.std(strip.astype(float),axis=1).max())<12:
                    top=edge;break
    readers=[r for r in regions if 'DEAR READERS' in text(r).upper() and r['bbox']['y']>h*.5]
    for r in readers:
        start=r['bbox']['y'];tail=[a for a in regions if a['bbox']['y']>=start]
        joined=' '.join(text(a).upper() for a in tail)
        if not ('UNLOCK UP TO' in joined and 'EXCLUSIVELY ON' in joined and any(vortex(a) for a in tail)):continue
        if any(not re.search(r'DEAR READERS|DIVE DEEPER|UNLOCK UP TO|EXCLUSIVELY ON|HAPPY.?READING|VORTEX',text(a),re.I) for a in tail):continue
        white=(reference.min(axis=2)>245).mean(axis=1)
        candidates=[y for y in range(max(top+1,int(start-w*.3)),int(start)) if np.all(white[y:y+12]>.99)]
        if candidates:bottom=min(bottom,min(candidates));break
    header=[r for r in regions if r['bbox']['y']<w*.65]
    header_text=' '.join(text(r) for r in header).upper()
    if ('THIS CHAPTER WAS BROUGHT TO' in header_text and 'VORTEX SCANS TEAM' in header_text
        and 'SHOW YOUR SUPPORT' in header_text and 'VORTEXSCANS.ORG' in header_text):
        unknown=[r for r in header if len(text(r))>8 and not re.search(r'VORTEX|SCANS|BROUGHT TO|SUPPORT|STORM AT|DISCORD',text(r),re.I)]
        if not unknown:
            last=max(r['bbox']['y']+r['bbox']['height'] for r in header)
            white=(reference.min(axis=2)>240).mean(axis=1)
            for edge in range(int(last),min(h-12,int(last+w*.2))):
                if np.all(white[edge:edge+12]>.98):top=max(top,edge);break
    # Explicit scanlator announcement cards end at a full-width dark gutter.
    # Require several independent ad anchors, not a generic word like MERGER.
    if ('MERGER' in header_text and 'ANNOUNCEMENT' in header_text
        and 'DISCORD' in header_text and 'RELEASES' in header_text
        and any(confirmed_credit(text(r)) for r in header)):
        last=max(r['bbox']['y']+r['bbox']['height'] for r in header)
        near_black=(reference.max(axis=2)<25).mean(axis=1)
        for edge in range(int(last),min(h-16,int(last+w*.3))):
            if np.all(near_black[edge:edge+16]>.99):
                top=edge;break
    if warnings:
        warning_y=min(r['bbox']['y'] for r in warnings)
        # Warning cards are often followed by a second illustrated scanlator
        # banner. Search through both stacked cards so the crop reaches the
        # real white/title-page boundary instead of leaving the lower ad.
        limit=min(h*.4,warning_y+w*2.1)
        anchors=[r for r in regions if warning_y<=r['bbox']['y']<limit and PROMO.search(text(r))]
        if len(anchors)>=5 and any(warning_y<=r['bbox']['y']<limit for r in domains):
            last=max(r['bbox']['y']+r['bbox']['height'] for r in anchors)
            for y in range(int(last),min(h-12,int(last+w*.35))):
                if dark[max(0,y-8):y].mean()>.92 and dark[y:y+8].mean()<.12:
                    top=y;break
            if not top:
                # Some RAWs append the warning card to a black story gutter.
                # Require a clear gray-card -> almost-black full-width edge.
                black=(reference.max(axis=2)<8).mean(axis=1)
                for y in range(int(last)+12,min(h-16,int(last+w*.35))):
                    if np.all(black[y:y+16]>.98) and black[y-12:y-4].mean()<.2:
                        top=y;break
            if not top:
                # A colored separator line can sit between a dark ad stack and
                # the white title page. In that case the immediate dark->light
                # transition above is interrupted even though a stable, nearly
                # full-width white page begins just below it.
                white=(reference.min(axis=2)>240).mean(axis=1)
                for y in range(int(last)+4,min(h-16,int(last+w*.50))):
                    if np.all(white[y:y+16]>.985):
                        top=y;break
            if not top:
                # Thunder's anti-scraper warning sometimes joins directly to
                # the comic with no uniform separator. Multiple promo anchors,
                # the WARNING heading, and its own domain make the final promo
                # text edge a safe structural boundary.
                joined=' '.join(text(r) for r in anchors)
                if re.search(r'(?:EN[-\s]*)?THUNDER\s*SCANS?\.COM',joined,re.I):
                    top=min(h-1,int(last)+round(w*.055))
    starts=[r for r in regions if re.search(r'READ\s+ONLY\s+AT',text(r),re.I) and r['bbox']['y']>h*.5]
    for r in starts:
        y=int(r['bbox']['y'])
        tail=[p for p in regions if p['bbox']['y']>=y]
        if not (any('DISCORD' in text(p).upper() for p in tail) and any(confirmed_credit(text(p)) for p in tail)):
            continue
        candidates=[]
        for edge in range(max(top+1,y-round(w*.25)),y):
            if edge>=8 and dark[edge:edge+8].mean()>.75 and dark[edge-8:edge].mean()<.15:
                candidates.append(edge)
        if candidates:bottom=max(candidates,key=lambda edge:float(dark[edge]-dark[edge-1]));break
    # MistScans often appends one large illustrated recommendation card after
    # the final story panel. Two brand anchors plus its explicit advertising
    # copy distinguish this from an ordinary in-panel watermark. Cut at the
    # white-gutter -> artwork edge, preserving the complete comic above it.
    mist_tail=[r for r in regions if r['bbox']['y']>h*.82
               and re.search(r'MIST\s*SCANS?',text(r),re.I)]
    mist_copy=' '.join(text(r) for r in regions if r['bbox']['y']>h*.82).upper()
    if (len(mist_tail)>=2 and 'READ ON OUR SITE' in mist_copy
            and ('CHECK OTHER SERIES' in mist_copy or 'FASTEST RELEASES' in mist_copy)):
        first=min(int(r['bbox']['y']) for r in mist_tail)
        white=(reference.min(axis=2)>245).mean(axis=1)
        for edge in range(max(top+1,first-round(w*.55)),first):
            if (edge>=16 and np.all(white[edge-16:edge]>.985)
                    and float(np.mean(white[edge:edge+16]))<.55):
                bottom=min(bottom,edge)
                break
    eva_tail=[r for r in regions if r['bbox']['y']>h*.60]
    eva_tail_text=' '.join(text(r) for r in eva_tail).upper()
    if (any(eva(r) for r in eva_tail) and 'DISCORD' in eva_tail_text
        and ('REACH OUT' in eva_tail_text or ('READ' in eva_tail_text and 'ONLY AT' in eva_tail_text))):
        starts_eva=[r for r in eva_tail if re.search(r'READ|ONLY\s+AT',text(r),re.I)]
        if starts_eva:
            start=min(int(r['bbox']['y']) for r in starts_eva)
            row_mean=reference[:,round(w*.03):round(w*.97)].mean(axis=(1,2))
            begin=max(top+8,start-round(w*.65))
            candidates=[]
            for edge in range(begin,start-8):
                before=float(np.mean(row_mean[max(0,edge-6):edge]))
                after=float(np.mean(row_mean[edge:edge+6]))
                candidates.append((abs(after-before),edge))
            if candidates and max(candidates)[0]>=18:
                bottom=min(bottom,max(candidates)[1])
    recruitment=[r for r in regions if 'RECRUITING' in text(r).upper() and r['bbox']['y']>h*.7]
    for r in recruitment:
        start=r['bbox']['y'];tail=' '.join(text(p) for p in regions if p['bbox']['y']>=start).upper()
        if 'POSITIONS' not in tail or 'DISCORD' not in tail or 'JOIN US' not in tail:continue
        credits=[p for p in regions if text(p).strip().upper()=='CAPCOM' and start-w*.3<p['bbox']['y']<start]
        if not credits:continue
        credit_end=max(p['bbox']['y']+p['bbox']['height'] for p in credits)
        black=(reference.max(axis=2)<25).mean(axis=1)
        candidates=[y for y in range(int(credit_end)+4,int(start)) if np.all(black[max(0,y-4):y+4]>.99)]
        if candidates:bottom=min(bottom,max(candidates));break
    if top>=bottom:raise RuntimeError('V5.2 invalid promo boundaries')
    return top,bottom


def promo_groups(regions,w,top,bottom):
    anchors=[r for r in regions if confirmed_credit(str(r.get('text',''))) and top<=r['bbox']['y']<bottom]
    groups=[];used=set()
    for anchor in anchors:
        if id(anchor) in used:continue
        b=anchor['bbox'];members=[anchor]
        for r in regions:
            rb=r['bbox']
            # Tile overlap may OCR the same watermark twice with different
            # spellings. Treat both confirmed domain readings as one mark.
            intersection=max(0,min(rb['x']+rb['width'],b['x']+b['width'])-max(rb['x'],b['x']))*max(0,min(rb['y']+rb['height'],b['y']+b['height'])-max(rb['y'],b['y']))
            union=rb['width']*rb['height']+b['width']*b['height']-intersection
            if r is not anchor and id(r) not in used and confirmed_credit(str(r.get('text',''))) and union>0 and intersection/union>.8:
                members.append(r);continue
            if id(r) in used or r is anchor or not promo_text(r.get('text','')):continue
            gap=max(0,rb['y']-(b['y']+b['height']),b['y']-(rb['y']+rb['height']))
            horizontal=rb['x']<b['x']+b['width'] and rb['x']+rb['width']>b['x']
            if gap<=max(12,b['height']*.5) and horizontal and top<=rb['y']<bottom:members.append(r)
        used.update(map(id,members));groups.append(members)
    return groups


# Upper bound (as a fraction of page width) for the source pill emblem's height.
# ThunderScans-style emblems measure ~0.15*w on an 800px page, so the original
# 0.13 cutoff rejected them as "cannot locate source pill emblem" and the whole
# mark was left untouched. 0.17 admits those emblems while staying well below a
# typical panel height, so unrelated purple artwork is still excluded.
PILL_EMBLEM_MAX_H = .17


# OCR often reads a scanlator pill's own emblem, wing or ornaments as a few
# characters of junk ('+', 'X', 'M', 'prni', 'EtE', '11', 'og'). Those regions
# are decoration on the mark, never dialogue, so they must not veto the pill
# replacement. The filter is positional rather than textual: the fragment must
# sit almost entirely inside the measured pill mask, which is where a mark's
# own decoration lives. Real callouts inside a translated bubble are excluded
# from `protected` long before this point by the translations grouping.
def _is_mark_debris(text):
    """True only for fragments that can never be story dialogue.

    Kept deliberately minimal and text-only: a region with no alphabetic
    content at all, or a single letter, is emblem noise ('+', 'X', '11').
    Anything else ("NO", "OK", "HEY", "prni") is left to the positional check
    in the collision loop, because guessing from short text alone would risk
    erasing a real callout.
    """
    t = str(text or '').strip()
    if not t:
        return False
    letters = re.sub(r'[^A-Za-z]', '', t)
    return not letters or len(letters) == 1


def pill_mask(reference,group):
    """Find the purple emblem/tip around OCR-confirmed scanlator-pill text."""
    h,w=reference.shape[:2]
    boxes=[r['bbox'] for r in group]
    x=min(b['x'] for b in boxes);y=min(b['y'] for b in boxes)
    right=max(b['x']+b['width'] for b in boxes);bottom=max(b['y']+b['height'] for b in boxes)
    x1=max(0,int(x-w*.16));x2=min(w,int(right+w*.12))
    y1=max(0,int(y-w*.06));y2=min(h,int(bottom+w*.06))
    crop=reference[y1:y2,x1:x2]
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    purple=((hsv[:,:,0]>=125)&(hsv[:,:,0]<=169)&(hsv[:,:,1]>100)&(hsv[:,:,2]>65)).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(purple)
    icons=[]
    for label in range(1,n):
        px,py,pw,ph,area=stats[label]
        if w*.025<=ph<=np.ceil(w*PILL_EMBLEM_MAX_H) and w*.02<=pw<=w*.5 and area>40 and x1+px<x and abs(y1+py+ph/2-(y+bottom)/2)<w*.045:
            icons.append((abs(x-(x1+px+pw)),label))
    if not icons:
        # Purple artwork can connect to the emblem at low saturation. Require
        # a compact, vertically enclosed saturated component before recovering it.
        for saturation in (160,180):
            compact=((hsv[:,:,0]>=125)&(hsv[:,:,0]<=169)&(hsv[:,:,1]>saturation)&(hsv[:,:,2]>65)).astype(np.uint8)
            _,candidate_labels,candidate_stats,_=cv2.connectedComponentsWithStats(compact)
            candidates=[]
            for label in range(1,len(candidate_stats)):
                px,py,pw,ph,area=candidate_stats[label]
                if (w*.025<=ph<=np.ceil(w*PILL_EMBLEM_MAX_H) and w*.02<=pw<=w*.16 and area>100
                    and py>3 and py+ph<crop.shape[0]-3 and x1+px<x
                    and x1+px+pw<=x+8 and abs(y1+py+ph/2-(y+bottom)/2)<w*.045):
                    candidates.append((abs(x-(x1+px+pw)),label))
            if candidates:
                labels,stats,icons=candidate_labels,candidate_stats,candidates
                n=len(stats)
                break
    if not icons:raise RuntimeError('V5.2 cannot locate source pill emblem')
    icon=min(icons)[1];ix,iy,iw,ih,_=stats[icon]
    # Include purple outline/tip at the right, but not unrelated colored art.
    active=(labels==icon)
    for label in range(1,n):
        px,py,pw,ph,area=stats[label]
        # A caption outline encloses the black pill, whereas purple/blue art
        # may satisfy the same color and size filters but has no dark interior.
        interior=crop[py:py+ph,px:px+pw]
        dark_interior=float((interior.max(axis=2)<85).mean()) if interior.size else 0
        if dark_interior>.5 and area>40 and ph>=max(8,ih*.18) and x1+px+pw>=right-8 and x1+px+pw<=right+w*.12 and py>=iy-4 and py+ph<=iy+ih+4:
            active|=labels==label
    yy,xx=np.where(active)
    points=np.column_stack((xx,yy)).astype(np.int32)
    # Text bbox is inside the source pill, including disconnected white glyphs.
    points=np.vstack((points,[[x-x1-3,y-y1-3],[right-x1+4,y-y1-3],[right-x1+4,bottom-y1+3],[x-x1-3,bottom-y1+3]])).astype(np.int32)
    local=np.zeros(crop.shape[:2],np.uint8)
    cv2.fillConvexPoly(local,cv2.convexHull(points),255)
    # Include antialiased purple edges/glow outside the saturated contour.
    local=cv2.dilate(local,np.ones((7,7),np.uint8))
    mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=local
    yy,xx=np.where(mask);bounds=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
    # Large genuine pills can exceed half the page. Bound their size against
    # the OCR-confirmed caption as well as the page, not a fixed 50% cutoff.
    # The measured extent includes the 7x7 dilation and the caption's own
    # right-hand pad, so allow for those or a correct pill is rejected by a
    # handful of pixels ("implausibly wide").
    caption_width=right-x
    edge_pad=round(w*.035)
    allowed_width=max(w*.5,min(w*.65,caption_width+min(w*.20,caption_width*.65)+16))+edge_pad
    if bounds[2]-bounds[0]>allowed_width:raise RuntimeError('V5.2 pill extent is implausibly wide')
    return mask,bounds


def text_mask(reference,group):
    mask=np.zeros(reference.shape[:2],np.uint8);h,w=mask.shape
    for r in group:
        b=r['bbox'];pad=max(5,min(12,round(b['height']*.15)))
        x,y,bw,bh=(int(b[k]) for k in ('x','y','width','height'))
        mask[max(0,y-pad):min(h,y+bh+pad),max(0,x-pad):min(w,x+bw+pad)]=255
    return mask


def eva_pill_mask(reference,group):
    """Cover EvaScans' portrait plus its small EVASCANS.ORG capsule."""
    h,w=reference.shape[:2]
    boxes=[r['bbox'] for r in group]
    x=min(int(b['x']) for b in boxes);y=min(int(b['y']) for b in boxes)
    right=max(int(b['x']+b['width']) for b in boxes)
    bottom=max(int(b['y']+b['height']) for b in boxes)
    # Eva's signature variants may have a portrait on the left and a second
    # round source emblem attached just beyond the capsule on the right.
    # Cover both satellites, not merely the OCR-visible domain in the middle.
    bounds=(max(0,x-round(w*.120)),max(0,y-round(w*.035)),
            min(w,right+round(w*.105)),min(h,bottom+round(w*.035)))
    x1,y1,x2,y2=bounds;crop=reference[y1:y2,x1:x2]
    if not crop.size:raise RuntimeError('V5.2 cannot locate EvaScans pill')
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    pink=(((hsv[:,:,0]>=145)|(hsv[:,:,0]<=8))&(hsv[:,:,1]>55)&(hsv[:,:,2]>45))
    dark=crop.max(axis=2)<115
    if float(pink.mean())<.018 or float(dark.mean())<.10:
        raise RuntimeError('V5.2 EvaScans pill color signature missing')
    mask=np.zeros((h,w),np.uint8);mask[y1:y2,x1:x2]=255
    return mask,bounds


def dusk_pill_mask(reference, group):
    """Cover DuskScans' red portrait medallion and attached dark capsule."""
    h, w = reference.shape[:2]
    boxes = [r['bbox'] for r in group]
    x = min(int(b['x']) for b in boxes); y = min(int(b['y']) for b in boxes)
    right = max(int(b['x'] + b['width']) for b in boxes)
    bottom = max(int(b['y'] + b['height']) for b in boxes)
    bounds = (max(0, x - round(w * .14)), max(0, y - round(w * .060)),
              min(w, right + round(w * .025)), min(h, bottom + round(w * .038)))
    x1, y1, x2, y2 = bounds; crop = reference[y1:y2, x1:x2]
    if not crop.size: raise RuntimeError('V5.2 cannot locate DuskScans pill')
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    red = (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 170))
           & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 45))
    dark = crop.max(axis=2) < 105
    if float(red.mean()) < .025 or float(dark.mean()) < .10:
        raise RuntimeError('V5.2 DuskScans pill color signature missing')
    mask = np.zeros((h, w), np.uint8); mask[y1:y2, x1:x2] = 255
    return mask, bounds


def mistscans_mark_mask(reference, group):
    """Cover MistScans' black wing emblem together with its domain caption."""
    h, w = reference.shape[:2]
    boxes = [r['bbox'] for r in group]
    x = min(int(b['x']) for b in boxes); y = min(int(b['y']) for b in boxes)
    right = max(int(b['x'] + b['width']) for b in boxes)
    bottom = max(int(b['y'] + b['height']) for b in boxes)
    # OCR sees the caption but normally misses the stylized wing on its left.
    # The mark is compact and its dimensions are stable relative to page width.
    bounds = (max(0, x - round(w * .075)), max(0, y - round(w * .028)),
              min(w, right + round(w * .018)), min(h, bottom + round(w * .028)))
    x1, y1, x2, y2 = bounds
    crop = reference[y1:y2, x1:x2]
    if not crop.size:
        raise RuntimeError('V5.2 cannot locate MistScans mark')
    # Avoid expanding an accidental OCR match over ordinary artwork: the
    # known signature contains a meaningful amount of dark emblem/text ink.
    if float((crop.max(axis=2) < 105).mean()) < .035:
        raise RuntimeError('V5.2 MistScans ink signature missing')
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask, bounds


def apply_branding_v52(image_path,ocr_path,logo_path,text_path,*,first_page=False,banner_path=None,inpaint=None,logo_layout=None,branding_version=VERSION):
    with Image.open(image_path) as opened:output=np.array(opened.convert('RGB'))
    page=Path(ocr_path).parent
    ocr=json.loads(Path(ocr_path).read_text(encoding='utf-8'));regions=ocr.get('regions',[])
    all_regions=list(regions)
    promotion_report=page/'typeset-report.json'
    removed_promo_ids=set();removed_promo_groups=set()
    if promotion_report.exists():
        for decision in json.loads(promotion_report.read_text()).get('automatic_promotions',[]):
            if decision.get('action')=='REMOVED_PROMOTION':
                removed_promo_ids.update(decision.get('source_ids',[]))
                removed_promo_groups.update(decision.get('group_ids',[]))
    regions=[r for r in regions if r.get('id') not in removed_promo_ids]
    pipeline_path=page/'pipeline-status.json'
    reference=output
    if pipeline_path.exists():
        source=json.loads(pipeline_path.read_text(encoding='utf-8')).get('input')
        if source:
            with Image.open(source) as opened:reference=np.array(opened.convert('RGB'))
    if reference.shape!=output.shape:raise RuntimeError('V5.2 requires the unbranded full-size render')
    h,w=output.shape[:2];top,bottom=card_bounds(reference,regions)
    madara_standalone=standalone_madara_promo(reference,all_regions)
    eva_standalone=standalone_eva_promo(reference,all_regions)
    if banner_path is not None and (standalone_promo(reference,all_regions) or madara_standalone or eva_standalone):
        with Image.open(banner_path) as opened:banner=opened.convert('RGB')
        banner=banner.resize((w,max(1,round(banner.height*w/banner.width))),Image.Resampling.LANCZOS)
        banner.save(image_path,format='PNG',optimize=True)
        return {'branding_engine':branding_version,'input_dimensions':[w,h],
                'output_dimensions':list(banner.size),'source_header_trimmed_at':h,
                'source_footer_trimmed_at':None,'standalone_promo_removed':True,
                'banner':[0,0,w,banner.height],'decisions':[],'replacements':[]}
    groups=promo_groups(regions,w,top,bottom)
    promo_ids={id(r) for g in groups for r in g}
    def nested_credit_fragment(r):
        b=r['bbox']
        if len(str(r.get('text','')).strip())>8:return False
        return any(b['x']>=a['bbox']['x'] and b['y']>=a['bbox']['y'] and b['x']+b['width']<=a['bbox']['x']+a['bbox']['width'] and b['y']+b['height']<=a['bbox']['y']+a['bbox']['height'] for group in groups for a in group)
    protected=[r for r in regions if id(r) not in promo_ids and not nested_credit_fragment(r)
               and not promo_text(r.get('text', '')) and top<=r['bbox']['y']<bottom
               # Short letter-less or 4-character OCR junk ('+', 'prni', 'X',
               # 'EtE', '11') is the source pill's own emblem/ornaments, not
               # story text. Keeping it in `protected` made the verified pill
               # veto its own decoration (a single touching pixel is enough)
               # and the original scanlator mark stayed on the page.
               and not _is_mark_debris(r.get('text', ''))]
    translations=json.loads((page/'translations.json').read_text()) if (page/'translations.json').exists() else {'groups':[]}
    dialogue_ids=set()
    for g in translations['groups']:
        if g['id'] in removed_promo_groups:continue
        # A scanlator pill's own emblem/ornaments OCR as tiny garbage ('+',
        # 'prni', 'X', '11'). The extractor cannot tell them from a callout, so
        # they arrive here as one-character "dialogue". Treating them as
        # protected text made the verified source pill collide with its own
        # decoration and the whole mark was left on the page.
        gtext=str(g.get('text','')).strip()
        if len(gtext)<=4 and not re.search(r'[A-Za-z]{3}',gtext):continue
        if not confirmed_credit(str(g.get('text',''))) and not promo_text(g.get('text','')):
            if top<=g['bbox']['y']<bottom:protected.append(g);dialogue_ids.add(g['id'])
    typeset_path=Path(image_path).parent/'typeset-report.json'
    if not typeset_path.exists():typeset_path=page/'typeset-report.json'
    if typeset_path.exists():
        clean_path=Path(image_path).parent/'cleaned.png'
        if not clean_path.exists():clean_path=page/'cleaned.png'
        cleaned_pixels=None
        if clean_path.exists():
            with Image.open(clean_path) as opened:cleaned_pixels=np.array(opened.convert('RGB'))
        for r in json.loads(typeset_path.read_text()).get('results',[]):
            if r.get('id') in dialogue_ids and r.get('bbox'):
                protected.append(measured_ink_region(r,output,cleaned_pixels))
    report={'branding_engine':branding_version,'input_dimensions':[w,h],'source_header_trimmed_at':top or None,
            'source_footer_trimmed_at':bottom if bottom<h else None,'decisions':[],'replacements':[],
            'text_watermark':None,'top_credit_cleared':None,'banner':None,'detected_credit_texts':[r['text'] for g in groups for r in g]}
    if inpaint is None and groups:
        from xianscan_client import XianScanClient
        inpaint=XianScanClient(os.getenv('XIANSCAN_URL','http://127.0.0.1:8123')).inpaint_mask
    with Image.open(logo_path) as opened:logo=opened.convert('RGBA')
    alpha=logo.getchannel('A').getbbox()
    if not alpha:raise RuntimeError('Empty brand logo')
    logo=logo.crop(alpha)
    if short_vortex_end_promo(reference,all_regions) or madara_standalone:
        canvas=Image.new('RGBA',(w,h),'white')
        target=round(w*.42)
        mark=logo.resize((target,max(1,round(logo.height*target/logo.width))),Image.Resampling.LANCZOS)
        pos=((w-mark.width)//2,(h-mark.height)//2)
        canvas.alpha_composite(mark,pos)
        image=canvas.convert('RGB')
        image.save(image_path,format='PNG',optimize=True)
        report['output_dimensions']=[w,h]
        report['standalone_promo_removed']=True
        promo_name='MADARASCANS FULL PROMO' if madara_standalone else 'VORTEX END PROMO'
        action='REPLACED_FULL_MADARASCANS_PROMO' if madara_standalone else 'REPLACED_SHORT_VORTEX_END_PROMO'
        mode='FULL_MADARASCANS_PROMO_REPLACEMENT' if madara_standalone else 'SHORT_VORTEX_END_PROMO_REPLACEMENT'
        report['standalone_promo_removed']=True
        report['decisions'].append({'text':promo_name,'action':action,'context':[0,0,w,h],'mask_pixels':w*h,'outside_mask_changed_pixels':0})
        report['replacements'].append({'logo':[pos[0],pos[1],mark.width,mark.height],'mode':mode,'opaque_background':False})
        return report
    for group in groups:
        combined=' '.join(r['text'] for r in group)
        if is_qimanga_group(group,locals().get('source','')):
            covered,rect=cover_qimanga(Image.fromarray(output),group,logo,top,bottom)
            output=np.array(covered)
            report['decisions'].append({'text':combined,'action':'INPAINTED_QIMANGA_REPLACEMENT','context':list(rect),'overlap_policy':'USER_APPROVED_FULL_COVER','opaque_background':False})
            report['replacements'].append({'mode':'TRANSPARENT_QIMANGA_REPLACEMENT','cleanup_rect':[rect[0],rect[1]-top,rect[2]-rect[0],rect[3]-rect[1]]})
            continue
        is_eva_pill=bool(re.search(r'EVASCANS|EVA\s+SCANS',combined,re.I)
                         and re.search(r'READ\s+THIS\s+SERIES\s+ONLY\s+AT',combined,re.I))
        is_dusk_pill=bool(re.search(r'DUSK\s*SCANS?(?:\.COM)?',combined,re.I))
        is_mistscans_mark=bool(re.search(r'MIST\s*SCANS?(?:\.COM)?',combined,re.I))
        # Tiny in-panel scanlator signatures are a capsule plus a separate
        # circular avatar/emblem. OCR usually sees only the domain, so treat
        # that compact domain as a pill and erase the complete visual mark.
        box_width=max(r['bbox']['x']+r['bbox']['width'] for r in group)-min(r['bbox']['x'] for r in group)
        box_height=max(r['bbox']['y']+r['bbox']['height'] for r in group)-min(r['bbox']['y'] for r in group)
        is_compact_brand_pill=bool(
            confirmed_credit(combined)
            and box_width <= w*.34 and box_height <= w*.09
            and re.search(r'(?:MADARA|EVA|SCAN|\.(?:COM|ORG))',combined,re.I)
        )
        is_pill=bool(is_eva_pill
                     or is_mistscans_mark
                     or is_compact_brand_pill
                     or re.search(r'(?:YOU|[YC]?OU)\s*CAN\s*READ\s*(?:THE\s+)?CHAPTER',combined,re.I)
                     or (confirmed_credit(combined) and re.search(r'\b(?:CH)?APTER\s+ON\b',combined,re.I)))
        if is_pill:
            try:
                if is_eva_pill:
                    mask,bounds=eva_pill_mask(reference,group)
                elif is_dusk_pill:
                    mask,bounds=dusk_pill_mask(reference,group)
                elif is_mistscans_mark:
                    mask,bounds=mistscans_mark_mask(reference,group)
                else:
                    mask,bounds=pill_mask(reference,group)
                if is_compact_brand_pill and not is_eva_pill and not is_dusk_pill:
                    # The round Madara avatar is frequently disconnected from
                    # the purple capsule, so component tracing can select only
                    # the bar. User-approved source replacement must erase the
                    # complete signature before placing Ryukomik branding.
                    boxes=[r['bbox'] for r in group]
                    sx=min(int(b['x']) for b in boxes);sy=min(int(b['y']) for b in boxes)
                    sr=max(int(b['x']+b['width']) for b in boxes)
                    sb=max(int(b['y']+b['height']) for b in boxes)
                    full=(max(0,sx-round(w*.14)),max(0,sy-round(w*.045)),
                          min(w,sr+round(w*.035)),min(h,sb+round(w*.045)))
                    fx1,fy1,fx2,fy2=full
                    mask[fy1:fy2,fx1:fx2]=255
                    bounds=full
            except RuntimeError as exc:
                if str(exc) not in ('V5.2 cannot locate source pill emblem', 'V5.2 pill extent is implausibly wide'):
                    raise
                # Uncertain branding geometry must not destroy dialogue or
                # abort an otherwise translated chapter. Leave this mark
                # untouched and record the limitation explicitly.
                report['decisions'].append({'text':combined,'action':'PRESERVED_UNCERTAIN_SOURCE_MARK',
                    'reason':str(exc),'mask_pixels':0,'outside_mask_changed_pixels':0})
                continue
        else:
            mask=text_mask(reference,group);yy,xx=np.where(mask);bounds=(int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1))
        if bounds[1]<top or bounds[3]>bottom:raise RuntimeError('V5.2 cleanup crosses a retained page boundary')
        collision=None
        for r in protected:
            b=r['bbox'];x,y,bw,bh=(int(b[k]) for k in ('x','y','width','height'))
            # Lightning emblems can OCR as "11". Ignore only tiny non-word
            # fragments mostly inside the measured emblem, left of its caption.
            core=mask[max(0,y):min(h,y+bh),max(0,x):min(w,x+bw)]
            # A confirmed MistScans domain and its wing are entirely source
            # branding. Duplicate/tiled OCR fragments inside that compact
            # rectangle must not veto the user-approved full replacement.
            if (is_mistscans_mark and core.size
                    and float(np.mean(core > 0)) > .65):
                continue
            # Dusk's tiny tagline and ornaments are frequently OCR'd as
            # gibberish dialogue. When that fragment is compact and almost
            # entirely enclosed by the verified red/black Dusk capsule, it is
            # part of the source mark rather than protected story text.
            if (is_dusk_pill and bh <= w*.06 and bw <= w*.32 and core.size
                    and float(np.mean(core>0)) > .92):
                continue
            if is_pill and not re.search(r'[A-Za-z]{2}',str(r.get('text',''))) and len(str(r.get('text','')))<=3 and x+bw<=min(a['bbox']['x'] for a in group) and core.size and float(np.mean(core>0))>.75:
                continue
            # The source pill's own emblem and ornaments OCR as short garbage
            # ('+', 'prni', 'EtE', '11', 'X') on either side of the caption.
            # When such a fragment is swallowed by the measured pill mask it IS
            # part of the source mark, not dialogue the operator protected;
            # letting it veto the replacement leaves the original scanlator
            # branding on the page. A 1-3 character, letter-less fragment is
            # never real dialogue, so mere contact with the mask is enough --
            # these are dots and strokes on the emblem itself.
            if is_pill and core.size and np.any(core > 0):
                rtext = str(r.get('text', '')).strip()
                if not re.search(r'[A-Za-z]{2}', rtext) and len(rtext) <= 3:
                    continue
                covered = float(np.mean(core > 0))
                # A mark fragment the pill genuinely swallows.
                if covered > .75 and not re.search(r'[A-Za-z]{3}', rtext):
                    continue
                # A neighbouring panel-art region that merely clips the mask
                # edge (a few percent of its area) is not inside the mark. A
                # real translatable callout sits mostly inside the bubble, so
                # this only drops accidental single-pixel contact.
                if covered < .15:
                    continue
            if intersects_protected(mask,(0,0),r):
                collision=r
                break
        if collision is not None:
            report['decisions'].append({'text':combined,'action':'PRESERVED_PROTECTED_OVERLAP',
                'context':list(bounds),'protected_text':str(collision.get('text',collision.get('id'))),
                'reason':'Source mark intersects protected text; no cleanup or logo applied',
                'mask_pixels':0,'outside_mask_changed_pixels':0})
            continue
        # Repair from RAW context, not remnants of the previous branding pass.
        try:
            repaired=inpaint(reference,mask)
        except Exception as exc:
            from requests.exceptions import HTTPError
            response=getattr(exc,'response',None)
            if not isinstance(exc,HTTPError) or response is None or response.status_code!=400:
                raise
            # A rejected optional branding mask must not discard all completed
            # story pages. Preserve this area and expose the service reason.
            report['decisions'].append({'text':combined,'action':'PRESERVED_REJECTED_SOURCE_MASK',
                'reason':response.text[:300],'mask_pixels':0,'outside_mask_changed_pixels':0})
            continue
        if repaired.shape!=reference.shape or np.any(np.any(repaired!=reference,axis=2)&(mask==0)):
            raise RuntimeError('V5.2 inpainting changed pixels outside source mask')
        output[mask>0]=repaired[mask>0]
        decision={'text':combined,'action':'REPLACED_PILL' if is_pill else 'REMOVED_SOURCE_TEXT','context':list(bounds),'mask_pixels':int(np.count_nonzero(mask)),'outside_mask_changed_pixels':0}
        if is_pill:
            x1,y1,x2,y2=bounds
            target=min(round(w*.20),x2-x1,round((y2-y1)*logo.width/logo.height))
            mark=logo.resize((target,max(1,round(logo.height*target/logo.width))),Image.Resampling.LANCZOS)
            pos=((x1+x2-mark.width)//2,(y1+y2-mark.height)//2)
            # Keep logo within the actual erased source silhouette.
            region=mask[pos[1]:pos[1]+mark.height,pos[0]:pos[0]+mark.width]
            while target>20 and np.any((np.array(mark.getchannel('A'))>0)&(region==0)):
                target-=2;mark=logo.resize((target,max(1,round(logo.height*target/logo.width))),Image.Resampling.LANCZOS)
                pos=((x1+x2-mark.width)//2,(y1+y2-mark.height)//2)
                region=mask[pos[1]:pos[1]+mark.height,pos[0]:pos[0]+mark.width]
            if target<=20:raise RuntimeError('V5.2 cannot fit replacement logo inside source pill')
            layout_details={}
            if logo_layout is not None:
                mark,pos,layout_details=logo_layout(logo,bounds,mask,output,protected)
            canvas=Image.fromarray(output).convert('RGBA');canvas.alpha_composite(mark,pos);output=np.array(canvas.convert('RGB'))
            report['replacements'].append({'logo':[pos[0],pos[1]-top,mark.width,mark.height],'mode':'SOURCE_PILL_REPLACEMENT',**layout_details})
        report['decisions'].append(decision)
    # The template match is source-specific and already collision guarded, so
    # it is safe across titles. Restricting it to one title allowed identical
    # Vortex emblems to survive in every other project whenever OCR missed the
    # stylized logo.
    from vortex_emblems import clean_emblems
    if inpaint is None:
        from xianscan_client import XianScanClient
        inpaint=XianScanClient(os.getenv('XIANSCAN_URL','http://127.0.0.1:8123')).inpaint_mask
    output,emblem_decisions=clean_emblems(reference,output,protected,top,bottom,inpaint)
    report['decisions'].extend(emblem_decisions)
    image=Image.fromarray(output[top:bottom]).convert('RGBA')
    if text_path is not None and Path(text_path).exists():
        from brand_postprocess import empty_white_position,resized
        with Image.open(text_path) as opened:watermark_text=opened.convert('RGBA')
        shifted=[]
        for item in protected:
            box=dict(item['bbox'])
            if box['y']+box['height']<=top or box['y']>=bottom:continue
            box['y']-=top
            shifted.append({'bbox':box})
        position=empty_white_position(image,watermark_text,shifted)
        if position:
            mark=resized(watermark_text,round(image.width*.56),opacity=1.0)
            image.alpha_composite(mark,position)
            report['text_watermark']=[position[0],position[1],mark.width,mark.height]
    if banner_path is not None:
        with Image.open(banner_path) as opened:banner=opened.convert('RGBA')
        banner=banner.resize((w,max(1,round(banner.height*w/banner.width))),Image.Resampling.LANCZOS)
        extended=Image.new('RGBA',(w,image.height+banner.height),'white');extended.alpha_composite(image);extended.alpha_composite(banner,(0,image.height))
        report['banner']=[0,image.height,w,banner.height];image=extended
    report['output_dimensions']=list(image.size)
    report['collision_guard']='measured-ink-v1'
    image.convert('RGB').save(image_path,format='PNG',optimize=True)
    return report
