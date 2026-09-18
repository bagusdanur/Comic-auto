"""V5: measured glyph fitting against a local, eroded surface mask.

Candidates are checked before compositing; text is never clipped to force a fit.
Font size takes precedence over wrapping aesthetics. No full-page masks needed.
"""
from __future__ import annotations

from functools import lru_cache
import math
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from source_text_color import source_text_color
from source_typography import source_typography,select_font
from enclosed_bubble import enclosed_bubble_mask
from outside_comment import comment_candidates


def fit_text_v5(text, font_path, mask, minimum=22, maximum=36, stroke=0, _hyphen_fallback=False, anchor_center=None, alignment='center'):
    text = text.translate(str.maketrans({'~':'...', '\u2019':"'", '\u2018':"'", '\u201c':'"', '\u201d':'"', '\u2014':'-', '\u2013':'-', '\u2026':'...'}))
    # Newlines supplied for system/skill windows are semantic separators, not
    # ordinary whitespace. Keep them as mandatory breaks while still wrapping
    # each section to the measured safe mask.
    paragraphs = [part.split() for part in text.splitlines()]
    paragraphs = [part for part in paragraphs if part]
    words = [word for part in paragraphs for word in part]
    if not words:
        return None
    height, width = mask.shape
    safe = mask.astype(bool)
    center_x,center_y=anchor_center or (width/2,height/2)
    center_x=min(max(float(center_x),0),width)
    center_y=min(max(float(center_y),0),height)
    for size in range(maximum, minimum - 1, -1):
        font = ImageFont.truetype(str(font_path), size)
        spacing = max(2, round(size * .12))
        probe = ImageDraw.Draw(Image.new('L', (1, 1)))

        def measure_words(sequence, a, b):
            line=' '.join(sequence[a:b])
            if _hyphen_fallback:line=line.replace('- ', '-')
            box = probe.textbbox((0, 0), line, font=font, stroke_width=stroke)
            return box[2] - box[0]

        candidates = []
        # Enumerate widths, then balance all word-preserving partitions via DP.
        # No language-dependent word truncation or arbitrary line-count cap.
        for budget in sorted(set([width - 4] + [round(width * f) for f in (.9, .8, .7, .6, .5)]), reverse=True):
            def wrap_paragraph(sequence):
              @lru_cache(None)
              def wrap(start):
                if start == len(sequence):
                    return 0., ()
                best = None
                for end in range(start + 1, len(sequence) + 1):
                    measured = measure_words(sequence, start, end)
                    if measured > budget:
                        break
                    rest = wrap(end)
                    if rest is None:
                        continue
                    penalty = (budget - measured) ** 2
                    if end == start + 1 and len(sequence[start]) < 4 and len(sequence) > 3:
                        penalty += budget ** 2
                    candidate = (rest[0] + penalty, (' '.join(sequence[start:end]),) + rest[1])
                    if best is None or candidate[0] < best[0]:
                        best = candidate
                return best
              return wrap(0)
            sections = [wrap_paragraph(sequence) for sequence in paragraphs]
            if any(section is None for section in sections):
                continue
            partition = (sum(section[0] for section in sections),
                         tuple(line for section in sections for line in section[1]))
            wrapped = '\n'.join(partition[1])
            if _hyphen_fallback:wrapped=wrapped.replace('- ', '-')
            bbox = probe.multiline_textbbox((0, 0), wrapped, font=font, spacing=spacing, align=alignment, stroke_width=stroke)
            w, h = math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1])
            if w > width or h > height:
                continue
            ink = Image.new('L', (w + 2, h + 2))
            ImageDraw.Draw(ink).multiline_text((1 - bbox[0], 1 - bbox[1]), wrapped, font=font, spacing=spacing, align=alignment, fill=255, stroke_width=stroke)
            pixels = np.asarray(ink) > 0
            # Preserve the visual centre of the RAW lettering. Bubble tails and
            # asymmetric panels can move the surface bounding-box centre far
            # away from the intended reading position, so explore locally
            # around the source anchor on both axes.
            for dx,dy in ((0,0),(0,-round(height*.06)),(0,round(height*.06)),
                          (-round(width*.04),0),(round(width*.04),0)):
                x=round(center_x-ink.width/2)+dx
                y=round(center_y-ink.height/2)+dy
                if x < 0 or y < 0 or x + ink.width > width or y + ink.height > height:
                    continue
                if np.any(pixels & ~safe[y:y + ink.height, x:x + ink.width]):
                    continue
                candidates.append((partition[0] + (abs(dx)+abs(dy))*width, {
                    'font': font, 'font_size': size, 'wrapped': wrapped,
                    'spacing': spacing, 'origin': (x + 1 - bbox[0], y + 1 - bbox[1]),
                    'line_count': len(partition[1]), 'overflow_pixels': 0,
                    'alignment': alignment,
                }))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]
    if not _hyphen_fallback and '-' in text:
        # Existing hyphens are legitimate line-break opportunities. Preserve
        # every character and retry without shrinking below the readable floor.
        return fit_text_v5(text.replace('-', '- '),font_path,mask,minimum,maximum,stroke,True,anchor_center,alignment)
    return None


def fit_constrained_text(text, font_path, mask, stroke=0, anchor_center=None, alignment='center'):
    """Last resort: readable floor, then a conservative syllable line break.

    Only split inside long alphabetic words, after a vowel and before a
    consonant-vowel syllable. Keep Indonesian consonant digraphs intact.
    Never truncate text, expand the safe mask, or go below 18 pixels.
    """
    fitted = fit_text_v5(text, font_path, mask, 18, 21, stroke, anchor_center=anchor_center, alignment=alignment)
    if fitted:
        fitted['fallback_reason'] = 'CONSTRAINED_TEXT_18PX_FLOOR'
        return fitted
    import re
    vowels = 'aiueoAIUEO'
    candidates = []
    for match in re.finditer(r'[^\W\d_]{8,}', text):
        word = match.group()
        for pos in range(3, len(word) - 2):
            suffix = word[pos:].lower()
            onset = 2 if suffix[:2] in ('ng', 'ny', 'sy', 'kh') else 1
            if word[pos-1] not in vowels or word[pos] in vowels:
                continue
            if len(suffix) <= onset or suffix[onset] not in vowels:
                continue
            split = match.start() + pos
            candidate = fit_text_v5(text[:split] + '- ' + text[split:], font_path, mask, 18, 21, stroke, True, anchor_center, alignment)
            if candidate and '-\n' in candidate['wrapped']:
                candidate['fallback_reason'] = 'CONSTRAINED_SYLLABLE_BREAK'
                candidates.append(candidate)
    return max(candidates, key=lambda item: item['font_size']) if candidates else None


def source_surface_mask(reference, raw_box, surface, other_boxes):
    """Local safe region: source-centred bubble interior or bounded artwork text."""
    enclosed=enclosed_bubble_mask(reference,raw_box,other_boxes)
    if enclosed is not None:return enclosed
    if surface.get('surface_type')=='BUBBLE_FLAT':
        from open_bubble import open_bubble_mask
        opened=open_bubble_mask(reference,raw_box,other_boxes)
        if opened is not None:
            (a,b,c,d),safe=opened
            safe=cv2.erode(safe,np.ones((15,15),np.uint8))
            for box in other_boxes:
                x,y,w,h=(int(box[k]) for k in ('x','y','width','height'))
                l,t,r,bt=max(0,x-a-6),max(0,y-b-6),min(c-a,x+w-a+6),min(d-b,y+h-b+6)
                if r>l and bt>t:safe[t:bt,l:r]=0
            yy,xx=np.where(safe)
            if len(xx):
                l,t,r,bt=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
                return (a+l,b+t,a+r,b+bt),safe[t:bt,l:r]
    x, y, w, h = (int(raw_box[k]) for k in ('x', 'y', 'width', 'height'))
    page_h, page_w = reference.shape[:2]
    flat = surface.get('surface_type') == 'BUBBLE_FLAT'
    pad_x = round(w * (.30 if flat else .08))
    pad_y = max(16, round(h * (.9 if flat else .22)))
    x1, y1, x2, y2 = max(0, x-pad_x), max(0, y-pad_y), min(page_w, x+w+pad_x), min(page_h, y+h+pad_y)
    mask = np.ones((y2-y1, x2-x1), dtype=np.uint8) * 255
    if flat:
        crop = reference[y1:y2, x1:x2]
        # White interior constraint preserves oval curves and spiky boundaries.
        mask = ((crop.min(axis=2) > 205) & (crop.max(axis=2)-crop.min(axis=2) < 40)).astype(np.uint8)*255
        # Fill only original text footprint. The surrounding border stays intact.
        mask[y-y1:y+h-y1, x-x1:x+w-x1] = 255
        if surface.get('safe_polygon'):
            polygon = np.round(np.array(surface['safe_polygon']) - [x1,y1]).astype(np.int32)
            polygon_mask = np.zeros_like(mask)
            cv2.fillPoly(polygon_mask, [polygon], 255)
            mask &= polygon_mask
    for box in other_boxes:
        bx, by, bw, bh = (int(box[k]) for k in ('x','y','width','height'))
        a,b,c,d = max(0,bx-x1-6), max(0,by-y1-6), min(x2-x1,bx+bw-x1+6), min(y2-y1,by+bh-y1+6)
        if c>a and d>b:
            mask[b:d,a:c] = 0
    mask = cv2.erode(mask, np.ones((5,5),np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return (x1,y1,x2,y2), mask


def symmetric_outline_mask(fill_mask,stroke):
    """Expand the exact glyph mask equally in every direction."""
    if stroke <= 0:
        return fill_mask.copy()
    fill_array=np.asarray(fill_mask,dtype=np.uint8)
    kernel=np.ones((stroke*2+1,stroke*2+1),dtype=np.uint8)
    return Image.fromarray(cv2.dilate(fill_array,kernel,iterations=1),mode='L')


def paint_fitted_text(output,bounds,fitted,paint,stroke,outline,rotation=0.0,rotation_center=None,safe_mask=None):
    """Composite solid/gradient glyphs without painting outside measured ink."""
    x1,y1,x2,y2=bounds;size=(x2-x1,y2-y1)
    origin=(fitted['origin'][0],fitted['origin'][1])
    fill_mask=Image.new('L',size)
    ImageDraw.Draw(fill_mask).multiline_text(origin,fitted['wrapped'],font=fitted['font'],spacing=fitted['spacing'],align=fitted.get('alignment','center'),fill=255)
    if rotation:
        fill_mask=fill_mask.rotate(
            -rotation,resample=Image.Resampling.BICUBIC,expand=False,
            center=rotation_center or (size[0]/2,size[1]/2),
        )
    outline_mask=symmetric_outline_mask(fill_mask,stroke)
    if safe_mask is not None and np.any((np.asarray(outline_mask)>0) & (safe_mask==0)):
        return False
    crop=output.crop(bounds).convert('RGBA')
    if stroke:
        # Never render the multiline block a second time with stroke_width.
        # Pillow can advance stroked multiline baselines differently, making
        # the outline drift farther down on every line. Dilating the exact fill
        # mask creates a truly symmetric keyline with identical glyph origins.
        layer=Image.new('RGBA',size,outline+(255,));layer.putalpha(outline_mask);crop.alpha_composite(layer)
    if paint.get('fill_mode')=='linear_gradient':
        start=np.array(paint['gradient_start'],dtype=float);end=np.array(paint['gradient_end'],dtype=float)
        length=size[1] if paint.get('gradient_axis')=='vertical' else size[0]
        ramp=np.linspace(0,1,max(1,length))[:,None]
        colors=np.clip(start+(end-start)*ramp,0,255).astype(np.uint8)
        rgb=np.repeat(colors[:,None,:],size[0],axis=1) if paint.get('gradient_axis')=='vertical' else np.repeat(colors[None,:,:],size[1],axis=0)
        layer=Image.fromarray(rgb,'RGB').convert('RGBA')
    else:
        layer=Image.new('RGBA',size,tuple(paint['color'])+(255,))
    layer.putalpha(fill_mask);crop.alpha_composite(layer);output.paste(crop.convert('RGB'),(x1,y1))
    return True


def render_page(args):
    reference=np.array(Image.open(args.reference_image).convert('RGB'))
    output=Image.open(args.cleaned_image).convert('RGB')
    cleaned_reference=np.array(output)
    if (reference.shape[1],reference.shape[0]) != output.size:
        raise ValueError('V5 reference/cleaned dimensions differ')
    groups=json.loads(Path(args.translations_json).read_text())['groups']
    retained_top,retained_bottom=0,reference.shape[0]
    if getattr(args,'ocr_json',None):
        from branding_v52 import card_bounds
        retained_top,retained_bottom=card_bounds(reference,json.loads(Path(args.ocr_json).read_text())['regions'])
    surfaces={s['id']:s for s in json.loads(Path(args.surface_map).read_text())['surfaces']}
    removal={s['id']:s for s in json.loads(Path(args.removal_report).read_text())['results']}
    from automatic_promotions import remove_promotions
    promo_regions=json.loads(Path(args.ocr_json).read_text()).get('regions',[]) if getattr(args,'ocr_json',None) else []
    promo_regions=[r for r in promo_regions if retained_top<=r['bbox']['y']<retained_bottom]
    processed,promo_groups,promo_decisions=remove_promotions(reference,np.array(output),promo_regions,groups)
    output=Image.fromarray(processed)
    results=[]
    for group in groups:
        surface=surfaces.get(group['id'],{})
        cleaned=removal.get(group['id'],{})
        b=group['bbox']
        if group['id'] in promo_groups:
            results.append({'id':group['id'],'bbox':b,'status':'REMOVED_PROMO','reason':'CONFIRMED_SOURCE_PROMOTION'})
            continue
        if b['y']+b['height']<=retained_top or b['y']>=retained_bottom:
            results.append({'id':group['id'],'bbox':b,'status':'PRESERVED','reason':'CONFIRMED_PROMO_PENDING_CROP'})
            continue
        if cleaned.get('status')=='PRESERVED' or surface.get('source_credit') or group['text'].strip().casefold()==group['translation'].strip().casefold():
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            output.paste(Image.fromarray(reference[y:y+h,x:x+w]),(x,y))
            results.append({'id':group['id'],'bbox':b,'status':'PRESERVED','reason':'SOURCE_PRESERVED'})
            continue
        if cleaned.get('status') not in ('CLEANED','INPAINTED'):
            raise RuntimeError(f"V5 cleanup incomplete: {group['id']}")
        anchor=b
        members=sorted([r for r in promo_regions if r.get('id') in group.get('source_region_ids',[])],key=lambda r:r['bbox']['y'])
        from typeset_test import source_rotation
        rotation=source_rotation(members)
        if len(members)>=4:
            def caps(r):
                letters=[c for c in r.get('text','') if c.isalpha()]
                return sum(c.isupper() for c in letters)/max(1,len(letters))
            head=members[0]['bbox'];body=[r['bbox'] for r in members[1:]]
            if caps(members[0])<.65 and all(caps(r)>.85 for r in members[1:]) and head['height']>1.4*np.median([r['height'] for r in body]) and head['y']+head['height']<=min(r['y'] for r in body):
                x=min(r['x'] for r in body);y=min(r['y'] for r in body)
                anchor=dict(x=x,y=y,width=max(r['x']+r['width'] for r in body)-x,height=max(r['y']+r['height'] for r in body)-y)
        bounds,mask=source_surface_mask(reference,anchor,surface,[g['bbox'] for g in groups if g['id']!=group['id']])
        if group.get('layout_constraint')=='raw_footprint':
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            y+=int(group.get('layout_offset_y',0))
            bounds=(x,y,x+w,y+h)
            mask=np.full((h,w),255,np.uint8)
            mask=cv2.erode(mask,np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        flat=surface.get('surface_type')=='BUBBLE_FLAT'
        # Source-relative scale, bounded for consistent readability. The old
        # 4% page-width target stopped at 32 px on standard 800 px webtoons,
        # leaving visibly unused balloon space. Use the RAW line footprint as
        # an additional signal while retaining the strict geometry fit.
        raw_line_height=(float(np.median([r['bbox']['height'] for r in members])) if members
                         else float(b['height'])/max(1,len(group.get('source_region_ids',[]))))
        preferred=min(args.max_font_size,max(28,round(output.width*.045),round(raw_line_height*.42)))
        # Narration over artwork needs only a clean keyline. Using the same
        # 2 px outline as SFX/shouts makes Anime Ace counters and corners look
        # swollen, especially on smaller captions.
        stroke=0 if flat else (1 if group.get('style')=='narration' else 2)
        style=source_text_color(reference,cleaned_reference,b)
        color=style['color']
        typography=source_typography(reference,cleaned_reference,b,group.get('style','dialogue'))
        selected_font=select_font(args,typography)
        system_markers=sum(bool(__import__('re').search(pattern,group.get('text',''),__import__('re').I)) for pattern in (
            r'\bRANK\s*:',r'\bDESCRIPTION\s*:',r'\bMATERIAL\b',r'\bEQUIPMENT\b',r'\bSKILL\b'))
        is_system_window=(system_markers>=2 or bool(
            __import__('re').match(r'\s*DESCRIPTION\s*:',group.get('text',''),__import__('re').I)
            and '\n' in group.get('translation','')
        ))
        alignment='left' if is_system_window else 'center'
        if is_system_window:
            selected_font=Path(args.font_dialogue or args.font)
            typography={**typography,'font_variant':'regular','style_source':'SYSTEM_WINDOW','style_confidence':1.0}
        if flat and sum(color)>600:
            # Preserve white RAW fill without making it invisible on white
            # balloons. Include the contrast outline in geometry fitting.
            stroke=2
        is_tiny_sfx = group.get('style') == 'sfx' and b['width'] * b['height'] < 2600
        def source_anchor(current_bounds):
            left,top,right,bottom=current_bounds
            return (b['x']+b['width']/2-left,b['y']+b['height']/2-top)
        anchor_center=source_anchor(bounds)
        fitted=fit_text_v5(group['translation'],selected_font,mask,max(22,args.min_font_size),preferred,stroke,anchor_center=anchor_center,alignment=alignment)
        regular_font=args.font_dialogue or args.font
        if not fitted and not is_system_window and str(selected_font)!=str(regular_font):
            fallback=fit_text_v5(group['translation'],regular_font,mask,max(22,args.min_font_size),preferred,stroke,anchor_center=anchor_center,alignment=alignment)
            if fallback:
                fitted=fallback;selected_font=regular_font
                typography={**typography,'font_variant':'regular','style_source':'FIT_FALLBACK','style_confidence':1.0}
        if not fitted and flat:
            stats=surface.get('stats',{})
            # Speed-line artwork is sometimes classified as a white balloon.
            # Retry only within the bounded RAW text neighborhood, preserving
            # other text exclusions, with an outline rather than a white card.
            if stats.get('light_ratio',1)<.8 and stats.get('dark_ratio',0)>.1:
                bounds,mask=source_surface_mask(reference,b,{**surface,'surface_type':'ARTWORK_TEXT'},[g['bbox'] for g in groups if g['id']!=group['id']])
                anchor_center=source_anchor(bounds)
                stroke=2
                fitted=fit_text_v5(group['translation'],selected_font,mask,max(22,args.min_font_size),preferred,stroke,anchor_center=anchor_center)
        if not fitted:
            # User-approved emergency floor for genuinely constrained text.
            # Reuse the same safe mask: never enlarge/ignore balloon boundaries.
            fitted=fit_constrained_text(group['translation'],selected_font,mask,stroke,anchor_center,alignment)
        if not fitted and not flat:
            # Perspective cards/SFX can receive an over-conservative inferred
            # polygon or collide with a neighbouring detection. The original
            # OCR footprint is always a valid last bounded placement area.
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            raw_mask=np.full((h,w),255,np.uint8)
            raw_mask=cv2.erode(raw_mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
            raw_fitted=fit_constrained_text(group['translation'],selected_font,raw_mask,stroke,(w/2,h/2))
            if not raw_fitted and stroke:
                raw_fitted=fit_constrained_text(group['translation'],selected_font,raw_mask,0,(w/2,h/2))
                if raw_fitted:
                    stroke=0
            if raw_fitted:
                bounds,mask,fitted=(x,y,x+w,y+h),raw_mask,raw_fitted
                anchor_center=(w/2,h/2)
                fitted['fallback_reason']='RAW_FOOTPRINT_18PX_FLOOR'
        if not fitted and is_tiny_sfx and stroke:
            fitted=fit_constrained_text(group['translation'],selected_font,mask,0,anchor_center)
            if fitted:
                stroke=0
        if not fitted and is_tiny_sfx:
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            output.paste(Image.fromarray(reference[y:y+h,x:x+w]),(x,y))
            results.append({'id':group['id'],'bbox':b,'status':'PRESERVED','reason':'TINY_SFX_TOO_CONSTRAINED'})
            continue
        if not fitted:
            # One pathological region must never discard every successfully
            # translated page in the chapter. Restore its exact RAW pixels;
            # future audit runs can improve it without corrupting the image.
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            output.paste(Image.fromarray(reference[y:y+h,x:x+w]),(x,y))
            results.append({'id':group['id'],'bbox':b,'status':'PRESERVED','reason':'UNFIT_SOURCE_RESTORED'})
            continue
        placement='surface-v5-small-font-fallback' if fitted.get('fallback_reason') else 'surface-v5'
        letters=[ch for ch in group['text'] if ch.isalpha()]
        comment_case=bool(letters) and sum(ch.isupper() for ch in letters)/len(letters)<.4
        exterior_candidates=comment_candidates(reference,cleaned_reference,b,[g['bbox'] for g in groups if g['id']!=group['id']]) if os.getenv('TYPESET_AUTO_COMMENT','1')=='1' and comment_case else []
        for candidate_bounds,candidate_mask in exterior_candidates:
            candidate=fit_text_v5(group['translation'],selected_font,candidate_mask,max(22,args.min_font_size),preferred,stroke)
            if candidate:
                bounds,mask,fitted=candidate_bounds,candidate_mask,candidate
                placement='automatic-exterior-comment-v1'
                break
        x1,y1,x2,y2=bounds
        ox,oy=fitted['origin']
        local=np.array(output)[y1:y2,x1:x2]
        dark=float(np.median(local))<100
        outline='black' if sum(color)>600 else 'white' if not dark else 'black'
        rotation_safe_mask=mask
        if rotation and surface.get('surface_type')=='TILTED_CARD':
            # Adjacent form fields make the unrotated exclusion mask extremely
            # tight at glyph corners. Permit only a small card-local keyline
            # gutter so the measured RAW angle survives without invading a
            # neighbouring field.
            rotation_gutter=min(24,max(8,round(
                min(mask.shape)*abs(math.sin(math.radians(rotation)))/2
            )))
            rotation_safe_mask=cv2.dilate(
                mask,np.ones((rotation_gutter*2+1,rotation_gutter*2+1),np.uint8),iterations=1
            )
        # fitted origin is local to bounds; paint through measured glyph masks.
        painted=paint_fitted_text(
            output,bounds,fitted,style,stroke,
            (0,0,0) if outline=='black' else (255,255,255),
            rotation=rotation,rotation_center=anchor_center,safe_mask=rotation_safe_mask,
        )
        if not painted and rotation:
            # A translated block is often wider than its RAW source. Preserve
            # the measured RAW angle by reducing size before giving up on the
            # rotation; never clip rotated glyphs against the safe surface.
            for rotated_size in range(fitted['font_size']-1,17,-1):
                rotated_fit=fit_text_v5(
                    group['translation'],selected_font,mask,
                    rotated_size,rotated_size,stroke,
                    anchor_center=anchor_center,
                )
                if not rotated_fit:
                    continue
                if paint_fitted_text(
                    output,bounds,rotated_fit,style,stroke,
                    (0,0,0) if outline=='black' else (255,255,255),
                    rotation=rotation,rotation_center=anchor_center,safe_mask=rotation_safe_mask,
                ):
                    fitted=rotated_fit
                    placement+='-rotation-fit'
                    painted=True
                    break
        if not painted and rotation and surface.get('surface_type')=='TILTED_CARD':
            # OCR boxes in forms commonly overlap the detector's larger row
            # region even though the visible label is safely contained. Keep
            # the rotation inside this group's own bounded render rectangle.
            painted=paint_fitted_text(
                output,bounds,fitted,style,stroke,
                (0,0,0) if outline=='black' else (255,255,255),
                rotation=rotation,rotation_center=anchor_center,
                safe_mask=np.full(mask.shape,255,np.uint8),
            )
            if painted:
                placement+='-tilted-card-bounds'
        if not painted:
            rotation=0.0
            paint_fitted_text(
                output,bounds,fitted,style,stroke,
                (0,0,0) if outline=='black' else (255,255,255),
            )
        results.append({'id':group['id'],'bbox':{'x':x1,'y':y1,'width':x2-x1,'height':y2-y1},'status':'TYPESET','layout_engine':'v5','placement_strategy':placement,'preferred_font_size':preferred,'font':str(selected_font),'rotation_degrees':rotation,**typography,**style,**{k:fitted[k] for k in ('font_size','wrapped','line_count','overflow_pixels')}})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    output.save(args.output)
    Path(args.report).write_text(json.dumps({'layout_engine':'v5','input_dimensions':list(output.size),'output_dimensions':list(output.size),'results':results,'automatic_promotions':promo_decisions},indent=2))
    return 0
