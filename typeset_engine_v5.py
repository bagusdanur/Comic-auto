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
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from source_text_color import source_text_color, harmonize_palette
from source_typography import source_typography,select_font,italic_run_reconciliation
from enclosed_bubble import enclosed_bubble_mask
from outside_comment import comment_candidates


def _vertical_band_center(safe, prefer):
    """Tengah pita baris mask yang lapang, dipakai kalau anchor mentok.

    Anchor dari glyph asli bisa duduk di tepi mask (dasar/dinding gelembung),
    sehingga blok teks besar tidak pernah muat walau ruang tengah gelembung
    lapang; mesin lalu mengecilkan font dan menempelkan teks ke atap bubble.
    Pita = baris berurutan yang lebarnya >=55% lebar maksimum, jadi ujung
    lancip dan ekor bubble tidak ikut digeser ke tengah.
    """
    if not safe.size:
        return None
    widths = safe.sum(axis=1)
    if not widths.any():
        return None
    usable = widths >= max(1, int(widths.max() * .55))
    idx = min(max(int(round(prefer)), 0), len(usable) - 1)
    if not usable[idx]:
        near = np.where(usable)[0]
        if not len(near):
            return None
        idx = int(near[np.argmin(np.abs(near - idx))])
    top = idx
    while top > 0 and usable[top - 1]:
        top -= 1
    bottom = idx
    while bottom < len(usable) - 1 and usable[bottom + 1]:
        bottom += 1
    return (top + bottom + 1) / 2


def fit_text_v5(text, font_path, mask, minimum=22, maximum=36, stroke=0, _hyphen_fallback=False, anchor_center=None, alignment='center', _balance_probe=False):
    text = text.translate(str.maketrans({'~':'...', '\u2019':"'", '\u2018':"'", '\u201c':'"', '\u201d':'"', '\u2014':'-', '\u2013':'-', '\u2026':'...'}))
    # Newlines supplied for system/skill windows are semantic separators, not
    # ordinary whitespace. Keep them as mandatory breaks while still wrapping
    # each section to the measured safe mask.
    paragraphs = [part.split() for part in text.splitlines()]
    # Kata majemuk panjang ("NGOMONG-NGOMONG...") tidak bisa dibungkus dan
    # memaksa font mengecil. Jadikan tanda hubung sebagai peluang baris baru.
    long_limit = max(6, int(mask.shape[1] / max(1, maximum * .55)))
    expanded = []
    for part in paragraphs:
        tokens = []
        for token in part:
            if '-' in token and len(token) > long_limit:
                chunk = ''
                for ch in token:
                    chunk += ch
                    if ch == '-' and chunk != token:
                        tokens.append(chunk)
                        chunk = ''
                if chunk:
                    tokens.append(chunk)
            else:
                tokens.append(token)
        expanded.append(tokens)
    paragraphs = [part for part in expanded if part]
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
        spacing = max(1, round(size * .05))
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
                    # Compare partitions across budgets without rewarding narrow columns.
                    penalty = ((budget - measured) / max(1,budget)) ** 2 * width ** 2
                    penalty += width ** 2 * .12
                    if end==len(sequence) and end==start+1 and len(sequence)>3:
                        penalty += width ** 2 * .35
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
            band_center = _vertical_band_center(safe, center_y)
            centers_y = [center_y]
            if band_center is not None and abs(band_center - center_y) > 2:
                centers_y.append(band_center)
            for band_y in centers_y:
              for dx,dy in ((0,0),(0,-round(height*.06)),(0,round(height*.06)),
                          (-round(width*.04),0),(round(width*.04),0)):
                x=round(center_x-ink.width/2)+dx
                y=round(band_y-ink.height/2)+dy
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
            best=min(candidates, key=lambda item: item[0])[1]
            if not _balance_probe and best['line_count']>=5 and size>minimum:
                alternative=fit_text_v5(text,font_path,mask,max(minimum,size-3),size-1,stroke,_hyphen_fallback,anchor_center,alignment,True)
                if alternative and alternative['line_count']<=best['line_count']-2:
                    alternative['layout_balance_reason']='FEWER_LINES_WITHIN_3PX'
                    return alternative
            return best
    if not _hyphen_fallback and '-' in text:
        # Existing hyphens are legitimate line-break opportunities. Preserve
        # every character and retry without shrinking below the readable floor.
        return fit_text_v5(text.replace('-', '- '),font_path,mask,minimum,maximum,stroke,True,anchor_center,alignment)
    return None


def fit_mask_center(text,font_path,mask,minimum=22,maximum=36,stroke=0,alignment='center'):
    """Fit di tengah area aman, bukan di titik tengah glyph sumber.

    Anchor dari glyph asli bisa jatuh di tepi mask: dasar gelembung sering
    terpotong artwork atau label tetangga, sehingga blok teks tidak pernah muat
    walau mask-nya lapang (pyscho ch1 page-005 bubble-002: anchor y=258.5 di
    mask setinggi 268, jadi hanya ~10 px yang tersisa di bawahnya). Tengah bbox
    piksel aman selalu berada di dalam area yang bisa dipakai. Hanya dipakai
    setelah semua jalur normal gagal, jadi tata letak yang sudah pas tidak
    bergeser.
    """
    ys,xs=np.where(mask>0)
    if not len(xs):
        return None
    candidates=[((float(xs.min())+float(xs.max())+1)/2,(float(ys.min())+float(ys.max())+1)/2)]
    centroid=(float(xs.mean()),float(ys.mean()))
    if abs(centroid[0]-candidates[0][0])>2 or abs(centroid[1]-candidates[0][1])>2:
        candidates.append(centroid)
    for center in candidates:
        fitted=fit_text_v5(text,font_path,mask,minimum,maximum,stroke,anchor_center=center,alignment=alignment)
        if fitted:
            fitted['fallback_reason']='MASK_CENTER_ANCHOR'
            return fitted
    return None


def largest_inscribed_rect(mask: np.ndarray) -> np.ndarray:
    """Persegi terbesar yang muat di dalam mask (histogram stack)."""
    m = (mask > 0).astype(np.uint8)
    height, width = m.shape
    heights = np.zeros(width, np.int32)
    best = (0, 0, 0, 0, 0)
    for y in range(height):
        heights = np.where(m[y] > 0, heights + 1, 0).astype(np.int32)
        stack = []
        for x in range(width + 1):
            current = int(heights[x]) if x < width else 0
            start = x
            while stack and stack[-1][1] > current:
                sx, sh = stack.pop()
                area = sh * (x - sx)
                if area > best[0]:
                    best = (area, sx, y - sh + 1, x, y + 1)
                start = sx
            if not stack or stack[-1][1] < current:
                stack.append((start, current))
    area, x0, y0, x1, y1 = best
    out = np.zeros_like(m)
    if area > 0:
        out[y0:y1, x0:x1] = 255
    return out


def polygon_is_rectangular(polygon, tolerance: float = 0.04) -> bool:
    """True hanya untuk panel yang benar-benar persegi empat sisi."""
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < 4:
        return False
    perimeter = float(cv2.arcLength(pts, True))
    if perimeter <= 0:
        return False
    approx = cv2.approxPolyDP(pts, tolerance * perimeter, True)
    return len(approx) == 4


def polygon_is_chat_panel(polygon, tolerance: float = 0.04) -> bool:
    """Panel chat asli: persegi empat sisi, terisi penuh, sumbu mendatar/tegak.

    Balon bertaji juga bisa di-approximate jadi 4 titik, jadi bentuk saja tidak
    cukup: rasio isi terhadap kotak pembatas harus mendekati penuh dan sisinya
    sejajar sumbu.
    """
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < 4 or not polygon_is_rectangular(polygon, tolerance):
        return False
    contour = pts.reshape(-1, 1, 2)
    area = float(cv2.contourArea(contour))
    (_, _), (w, h), angle = cv2.minAreaRect(contour)
    if w <= 1 or h <= 1:
        return False
    fill = area / (w * h)
    axis_offset = abs(((float(angle) + 45.0) % 90.0) - 45.0)
    return fill >= 0.92 and axis_offset <= 6.0


def polygon_is_irregular(polygon) -> bool:
    """Balon bertaji/bergerigi punya banyak sudut; oval/persegi tidak."""
    pts = np.asarray(polygon, dtype=np.float32)
    if len(pts) < 4:
        return False
    perimeter = float(cv2.arcLength(pts, True))
    if perimeter <= 0:
        return False
    approx = cv2.approxPolyDP(pts, 0.02 * perimeter, True)
    return len(approx) > 6


def outline_vertical_bounds(gray, box):
    """Cari tepi atas/bawah balon (garis gelap) di sekitar blok teks.

    `gray` adalah citra bersih (tanpa teks) dalam koordinat lokal mask, `box`
    adalah (x0, y0, x1, y1) blok teks. Pengamatan hanya pada kolom tengah blok
    supaya tidak tertipu artwork di sisi balon.
    """
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = max(0, x0), min(w - 1, x1)
    y0, y1 = max(0, y0), min(h - 1, y1)
    if x1 <= x0:
        return None
    centre_start = x0 + int((x1 - x0) * 0.35)
    centre_end = x0 + int((x1 - x0) * 0.65) + 1
    column = gray[:, centre_start:centre_end]
    dark_rows = np.where((column < 115).mean(axis=1) > 0.25)[0]
    above = dark_rows[dark_rows < y0]
    below = dark_rows[dark_rows > y1]
    if not len(above) or not len(below):
        return None
    return int(above.max()), int(below.min())


def recenter_fitted_block(fitted, mask, target=None):
    """Geser blok teks agar pusat tinta tepat di pusat lingkaran terbesar mask.

    Anchor sumber (posisi teks Inggris) sering mepet ke satu sisi balon, jadi
    blok hasil terjemahan tampak tidak di tengah. Pergeseran hanya dipakai bila
    seluruh tinta tetap berada di dalam area aman.
    """
    if not fitted or not mask.any():
        return fitted
    wrapped = fitted.get('wrapped') or ''
    font = fitted.get('font')
    if not wrapped or font is None:
        return fitted
    spacing = fitted.get('spacing') or 0
    h, w = mask.shape
    ox, oy = fitted['origin']
    layer = Image.new('L', (w, h), 0)
    ImageDraw.Draw(layer).multiline_text((ox, oy), wrapped, font=font, spacing=spacing,
                                         align=fitted.get('alignment') or 'center', fill=255)
    ink = np.asarray(layer) > 0
    if not ink.any():
        return fitted
    ys, xs = np.nonzero(ink)
    if target is not None:
        cx, cy = float(target[0]), float(target[1])
    else:
        rows, cols = np.nonzero(mask > 0)
        cy = float(rows.min() + rows.max()) / 2.
        cx = float(cols.min() + cols.max()) / 2.
    dx = int(round(float(cx) - (xs.min() + xs.max()) / 2.))
    dy = int(round(float(cy) - (ys.min() + ys.max()) / 2.))
    safe = mask > 0
    for shift_x, shift_y in ((dx, dy), (0, dy), (dx, 0)):
        if shift_x == 0 and shift_y == 0:
            continue
        shifted = np.zeros_like(ink)
        src_y0, src_y1 = max(0, -shift_y), min(h, h - shift_y)
        src_x0, src_x1 = max(0, -shift_x), min(w, w - shift_x)
        if src_y0 >= src_y1 or src_x0 >= src_x1:
            continue
        shifted[src_y0 + shift_y:src_y1 + shift_y, src_x0 + shift_x:src_x1 + shift_x] = ink[src_y0:src_y1, src_x0:src_x1]
        if int(shifted.sum()) != int(ink.sum()):
            continue
        if np.any(shifted & ~safe):
            continue
        fitted = dict(fitted)
        fitted['origin'] = (ox + shift_x, oy + shift_y)
        fitted['recenter'] = (shift_x, shift_y)
        return fitted
    return fitted


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


def raw_glyph_style(reference, regions, strict=False):
    """Ambil warna dan italic huruf langsung dari piksel sumber.

    `source_text_color` bisa mengambil warna isi bubble saat huruf sumber gelap
    di balon bernuansa/gradien (pyscho ch2 page-004 bubble-006: warna tersampel
    (133,127,139) padahal hurufnya hitam). Di sini piksel huruf diambil dari
    ekor distribusi luminansi (minoritas ekstrem), lalu kemiringan centroid
    baris dipakai sebagai bukti italic.
    """
    colors, slopes = [], []
    for region in regions:
        box = region.get("bbox")
        if not box:
            continue
        x, y, w, h = (int(box[k]) for k in ("x", "y", "width", "height"))
        if w < 8 or h < 8:
            continue
        crop = reference[max(0, y):y + h, max(0, x):x + w]
        if not crop.size:
            continue
        luminance = crop.mean(axis=2)
        median = float(np.median(luminance))
        background_is_light = median > 128
        if background_is_light:
            cut = max(median - 45, 60)
            ink = luminance < cut
        else:
            cut = min(median + 45, 200)
            ink = luminance > cut
        fraction = float(ink.mean())
        limit = .55 if strict else .40
        if fraction < .004 or fraction > limit or int(ink.sum()) < 40:
            continue
        ink_pixels = crop[ink]
        if strict and float(np.std(ink_pixels.reshape(-1, 3), axis=0).max()) > 85:
            continue
        colors.append(np.median(ink_pixels, axis=0))
        ys, xs = np.nonzero(ink)
        if len(ys) >= 80 and ys.std() >= 3:
            slopes.append(abs(float(np.polyfit(ys, xs, 1)[0])))
    if not colors:
        return None
    color = np.median(np.array(colors), axis=0)
    return {
        "color": [int(round(v)) for v in color],
        "italic": (float(np.median(slopes)) >= .18) if slopes else None,
    }


def _shape_score(polygon) -> float:
    """1.0 untuk persegi panjang (di sudut apa pun), ~0.785 untuk oval."""
    pts = np.asarray(polygon, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    area = abs(area) / 2
    bw = float(pts[:, 0].max() - pts[:, 0].min())
    bh = float(pts[:, 1].max() - pts[:, 1].min())
    if bw <= 0 or bh <= 0:
        return 0.0
    edges = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    longest = edges[int(np.argmax((edges ** 2).sum(axis=1)))]
    angle = abs(math.atan2(longest[1], longest[0]))
    angle = min(angle, math.pi - angle)
    return area / (bw * bh) * (math.cos(angle) + math.sin(angle)) ** 2


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


def fit_mask_size(mask, size):
    """Return the mask exactly `size`, cropping a surplus or padding a shortfall.

    Pillow rounds the expanded bounding box when it rotates: turning the
    transposed right-angle canvas (219,102) by exactly 90 degrees comes back
    as (103,220), one pixel over on each axis.  The RGBA layer below is built
    at `size` and the page-space guard is compared against `size`, so that
    surplus reached both and raised `operands could not be broadcast together
    with shapes (220,103) (219,102)` in the middle of Shadow Sovereign ch1.
    The extra pixel is bounding-box rounding, not glyph ink, so cropping it
    symmetrically loses nothing; a shortfall is padded with empty ink.
    """
    if mask.size == size:
        return mask
    fixed = Image.new('L', size)
    fixed.paste(mask, ((size[0] - mask.size[0]) // 2, (size[1] - mask.size[1]) // 2))
    return fixed


def symmetric_outline_mask(fill_mask,stroke):
    """Expand the exact glyph mask equally in every direction."""
    if stroke <= 0:
        return fill_mask.copy()
    fill_array=np.asarray(fill_mask,dtype=np.uint8)
    kernel=np.ones((stroke*2+1,stroke*2+1),dtype=np.uint8)
    return Image.fromarray(cv2.dilate(fill_array,kernel,iterations=1),mode='L')


def right_angle_fit_space(page_mask, anchor_center, bounds, rotation, footprint=None):
    """Map the page-space surface into the transposed canvas a 90 degree caption is fitted on.

    Returns the transposed mask plus the anchor in canvas coordinates. This is
    the exact inverse of the turn paint_fitted_text applies, so the finished
    glyphs land back inside the raw footprint; ``footprint`` (a group bbox)
    clips the fit first, because the surface a caption sits in can be far
    taller than its lettering and a long translation would otherwise sprawl
    onto the artwork and trip the QC spill guard. Anything that is not a
    right angle comes back untouched.
    """
    if not (rotation and abs(abs(rotation) - 90) < 1):
        return page_mask, anchor_center
    left, top, right, bottom = bounds
    page_w, page_h = right - left, bottom - top
    mask = page_mask
    if footprint is not None:
        clipped = np.zeros_like(mask)
        fx1 = max(0, int(footprint["x"]) - left)
        fx2 = min(page_w, int(footprint["x"]) + int(footprint["width"]) - left)
        fy1 = max(0, int(footprint["y"]) - top)
        fy2 = min(page_h, int(footprint["y"]) + int(footprint["height"]) - top)
        if fx2 > fx1 and fy2 > fy1:
            clipped[fy1:fy2, fx1:fx2] = 255
            mask = cv2.bitwise_and(mask, clipped)
    ax, ay = anchor_center
    if rotation < 0:
        # PIL turns this canvas +90 (counter-clockwise); the inverse maps a
        # page point (ax,ay) to (page_h-1-ay, ax).
        return np.rot90(mask, k=-1), (page_h - 1 - ay, ax)
    # PIL turns it -90 (clockwise); the inverse is (ay, page_w-1-ax).
    return np.rot90(mask, k=+1), (ay, page_w - 1 - ax)


def paint_fitted_text(output,bounds,fitted,paint,stroke,outline,rotation=0.0,rotation_center=None,safe_mask=None):
    """Composite solid/gradient glyphs without painting outside measured ink."""
    x1,y1,x2,y2=bounds;size=(x2-x1,y2-y1)
    origin=(fitted['origin'][0],fitted['origin'][1])
    # A right-angle caption is fitted against the *transposed* surface, because
    # its own footprint is tall and narrow and would shrink the translation to
    # an unreadable sliver. Draw on that transposed canvas and turn the
    # finished layer by exactly 90 degrees: expand=True maps the canvas back
    # onto the original bounds with no clipping, and the pivot is the canvas
    # centre -- which is what placed the anchor when the text was fitted.
    right_angle=bool(rotation) and abs(abs(rotation)-90)<1
    canvas=(size[1],size[0]) if right_angle else size
    fill_mask=Image.new('L',canvas)
    ImageDraw.Draw(fill_mask).multiline_text(origin,fitted['wrapped'],font=fitted['font'],spacing=fitted['spacing'],align=fitted.get('alignment','center'),fill=255)
    if rotation:
        if right_angle:
            pivot=(canvas[0]/2,canvas[1]/2)
        else:
            pivot=rotation_center or (size[0]/2,size[1]/2)
        fill_mask=fill_mask.rotate(
            -rotation,resample=Image.Resampling.BICUBIC,
            expand=right_angle,center=pivot,
        )
    fill_mask=fit_mask_size(fill_mask,size)
    outline_mask=symmetric_outline_mask(fill_mask,stroke)
    guard=np.asarray(safe_mask) if safe_mask is not None else None
    if guard is not None and guard.shape!=(size[1],size[0]):
        # bounds and the page-space surface are re-based together by the raw
        # footprint rescue and the exterior-comment placement. A caller that
        # moves one and not the other used to surface as an opaque numpy
        # broadcast error in the middle of a chapter, so name the invariant.
        raise RuntimeError(
            f"V5 glyph guard {guard.shape} does not match bounds {(size[1], size[0])}: "
            "re-point the page-space surface whenever bounds are re-based"
        )
    if guard is not None and np.any((np.asarray(outline_mask)>0) & (guard==0)):
        return False
    crop=output.crop(bounds).convert('RGBA')
    if paint.get('glow_radius_ratio',0)>0:
        radius=max(1,min(8,fitted.get('font_size',24)*paint['glow_radius_ratio']))
        glow=outline_mask.filter(ImageFilter.GaussianBlur(radius))
        alpha=np.asarray(glow).astype(float)*paint.get('glow_opacity',.35)
        if safe_mask is not None:alpha*=safe_mask>0
        layer=Image.new('RGBA',size,tuple(paint.get('glow_color',outline))+(255,))
        layer.putalpha(Image.fromarray(alpha.astype(np.uint8)));crop.alpha_composite(layer)
    if stroke:
        # Never render the multiline block a second time with stroke_width.
        # Pillow can advance stroked multiline baselines differently, making
        # the outline drift farther down on every line. Dilating the exact fill
        # mask creates a truly symmetric keyline with identical glyph origins.
        layer=Image.new('RGBA',size,outline+(255,));layer.putalpha(outline_mask);crop.alpha_composite(layer)
    if paint.get('fill_mode')=='linear_gradient':
        start=np.array(paint['gradient_start'],dtype=float);end=np.array(paint['gradient_end'],dtype=float)
        length=size[1] if paint.get('gradient_axis')=='vertical' else size[0]
        yy,xx=np.where(np.asarray(fill_mask)>0)
        coordinates=yy if paint.get('gradient_axis')=='vertical' else xx
        lo=int(coordinates.min()) if coordinates.size else 0
        hi=int(coordinates.max()) if coordinates.size else max(1,length-1)
        ramp=np.clip((np.arange(length)-lo)/max(1,hi-lo),0,1)[:,None]
        colors=np.clip(start+(end-start)*ramp,0,255).astype(np.uint8)
        rgb=np.repeat(colors[:,None,:],size[0],axis=1) if paint.get('gradient_axis')=='vertical' else np.repeat(colors[None,:,:],size[1],axis=0)
        layer=Image.fromarray(rgb,'RGB').convert('RGBA')
    else:
        layer=Image.new('RGBA',size,tuple(paint['color'])+(255,))
    layer.putalpha(fill_mask);crop.alpha_composite(layer);output.paste(crop.convert('RGB'),(x1,y1))
    return True


def restore_source_group(output, reference, group, regions, other_boxes):
    """Preserve rotated source lettering without restoring neighbours inside its bbox."""
    x,y,w,h=(int(group['bbox'][k]) for k in ('x','y','width','height'))
    x=max(0,x);y=max(0,y);w=min(w,reference.shape[1]-x);h=min(h,reference.shape[0]-y)
    members=[r for r in regions if r.get('id') in group.get('source_region_ids',[])]
    from translate_text_test import region_geometry_kind
    rotated=any(region_geometry_kind(r)=='rotated' for r in members)
    if not rotated:
        output.paste(Image.fromarray(reference[y:y+h,x:x+w]),(x,y));return
    mask=np.zeros((h,w),np.uint8)
    for region in members:
        polygon=region.get('polygon')
        if polygon:
            points=np.round(np.array(polygon)-np.array([x,y])).astype(np.int32)
            cv2.fillConvexPoly(mask,points,255)
    mask=cv2.dilate(mask,np.ones((5,5),np.uint8))
    if group.get('source_restore_mode')=='bbox_excluding_neighbors':mask[:]=255
    for box in other_boxes:
        left=max(0,int(box['x'])-x-2);top=max(0,int(box['y'])-y-2)
        right=min(w,int(box['x']+box['width'])-x+2);bottom=min(h,int(box['y']+box['height'])-y+2)
        if right>left and bottom>top:mask[top:bottom,left:right]=0
    output.paste(Image.fromarray(reference[y:y+h,x:x+w]),(x,y),Image.fromarray(mask))


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
    page_styles=harmonize_palette({g["id"]:source_text_color(reference,cleaned_reference,g["bbox"]) for g in groups})
    # Measure every group's RAW typography up front: the italic verdict can only
    # be reconciled across neighbouring bubbles before any font is chosen.
    page_typography={}
    for g in groups:
        entry=source_typography(reference,cleaned_reference,g['bbox'],g.get('style','dialogue'))
        entry['id']=g['id'];entry['bbox']=g['bbox']
        entry['surface_type']=(surfaces.get(g['id'],{}) or {}).get('surface_type')
        page_typography[g['id']]=entry
    italic_run_reconciliation(list(page_typography.values()))
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
        def _plain(value):
            return ''.join(ch for ch in str(value).casefold() if ch.isalnum())
        same_text = _plain(group['text']) == _plain(group['translation'])
        if cleaned.get('status')=='PRESERVED' or surface.get('source_credit') or same_text:
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            restore_source_group(output,reference,group,promo_regions,[g['bbox'] for g in groups if g['id']!=group['id']])
            results.append({'id':group['id'],'bbox':b,'status':'PRESERVED','reason':
                            'CONFIRMED_SOURCE_PROMOTION' if cleaned.get('reason') == 'CONFIRMED_SOURCE_PROMOTION'
                            else 'SOURCE_PRESERVED'})
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
        style=dict(page_styles[group["id"]])
        color=style['color']
        raw_crop=cleaned_reference[max(0,int(b['y'])):int(b['y']+b['height']),max(0,int(b['x'])):int(b['x']+b['width'])]
        if raw_crop.size and float(np.median(raw_crop))>190 and style.get('color_source')=='RAW_GLYPH_COLOR' and np.mean(color)>170:
            style['sampled_color']=style.get('sampled_color',color)
            factor=170/max(1,float(np.mean(color)))
            color=tuple(round(v*factor) for v in color);style['color']=color
            style['color_correction_reason']='PASTEL_CONTRAST_ON_LIGHT_SURFACE'
        if style.get('outline_source')=='RAW_GLYPH_RING':
            stroke=max(1,min(4 if sum(style.get('outline_color',(0,0,0)))>600 else 2,round(preferred*style['outline_width_ratio'])))
        typography=dict(page_typography.get(group["id"]) or source_typography(reference,cleaned_reference,b,group.get('style','dialogue')))
        # Warna & italic mengikuti huruf sumber. ARTWORK_TEXT hanya dipakai
        # kalau tinta hurufnya jelas minoritas dan seragam.
        surface_kind = surface.get('surface_type') or 'BUBBLE_FLAT'
        glyph_style = raw_glyph_style(reference, members,
                                      strict=(surface_kind == 'ARTWORK_TEXT'))
        if glyph_style:
            # Hanya perbaiki warna yang kontrasnya lemah terhadap isi bubble.
            # Kasus bug: huruf hitam di balon lavender tersampel abu-abu
            # (page-004 bubble-005: 136 vs isi 200). Kalau warna lama sudah
            # kontras (mis. putih di art gelap), jangan diubah.
            fill_luminance = None
            poly_fill = np.array(surface.get('safe_polygon') or [], dtype=np.int32)
            if len(poly_fill) >= 3:
                px0, py0 = poly_fill.min(axis=0)
                px1, py1 = poly_fill.max(axis=0) + 1
                fill_crop = cleaned_reference[max(0,int(py0)):int(py1), max(0,int(px0)):int(px1)]
                if fill_crop.size:
                    fill_mask = np.zeros(fill_crop.shape[:2], np.uint8)
                    cv2.fillPoly(fill_mask, [poly_fill - [max(0,int(px0)), max(0,int(py0))]], 255)
                    fill_pixels = fill_crop[fill_mask > 0]
                    if fill_pixels.size:
                        fill_luminance = float(np.median(fill_pixels.reshape(-1, 3).mean(axis=1)))
            old_luminance = float(np.mean(style['color']))
            use_sample = True
            if fill_luminance is not None:
                old_contrast = abs(old_luminance - fill_luminance)
                new_contrast = abs(float(np.mean(glyph_style['color'])) - fill_luminance)
                # Pilih kandidat dengan kontras lebih baik terhadap isi bubble;
                # warna lama dipertahankan kalau sudah jelas lebih kontras.
                use_sample = new_contrast > old_contrast + 10
            if use_sample:
                color = tuple(glyph_style['color'])
                style['color'] = color
                style['color_source'] = 'RAW_GLYPH_SAMPLE'
                style['color_confidence'] = 1.0
        # Lettering bersih: warna hasil sampling yang berwarna (hijau/biru) atau
        # abu-abu pudar terlihat jelek & tidak konsisten antar balon. Untuk non-SFX
        # snap ke hitam/putih kontras tinggi terhadap latar bersih setempat.
        # SFX dibiarkan artistik. Hitam/putih murni yang sudah ada tidak diubah.
        if group.get('style') != 'sfx':
            cy0=max(0,int(b['y']));cy1=min(cleaned_reference.shape[0],int(b['y']+b['height']))
            cx0=max(0,int(b['x']));cx1=min(cleaned_reference.shape[1],int(b['x']+b['width']))
            bg_crop=cleaned_reference[cy0:cy1,cx0:cx1]
            if bg_crop.size:
                bg_lum=float(np.median(bg_crop.reshape(-1,3).mean(axis=1)))
                cmean=float(np.mean(color));spread=float(max(color)-min(color))
                if spread>40 or 40<cmean<215:
                    snap=(0,0,0) if bg_lum>=128 else (255,255,255)
                    color=snap;style['color']=snap
                    style['color_source']='CLEAN_CONTRAST_SNAP'
                    style['outline_color']=(255,255,255) if snap==(0,0,0) else (0,0,0)
        # Kebijakan user: hanya font roman biasa. Italic DAN bold dimatikan.
        if str(typography.get('font_variant') or 'regular').lower() != 'regular':
            typography = {**typography, 'font_variant': 'regular',
                          'style_source': 'ROMAN_ONLY', 'style_confidence': 1.0}
        selected_font=select_font(args,typography)
        system_markers=sum(bool(__import__('re').search(pattern,group.get('text',''),__import__('re').I)) for pattern in (
            r'\bRANK\s*:',r'\bDESCRIPTION\s*:',r'\bMATERIAL\b',r'\bEQUIPMENT\b',r'\bSKILL\b'))
        is_system_window=(system_markers>=2 or bool(
            __import__('re').match(r'\s*DESCRIPTION\s*:',group.get('text',''),__import__('re').I)
            and '\n' in group.get('translation','')
        ))
        alignment='left' if is_system_window else 'center'
        # Lettering komik selalu kapital (sumbernya ALL CAPS). Teks sistem
        # (RANK:/DESCRIPTION:) dan bubble chat dibiarkan apa adanya.
        comic_text = group['translation']
        if not is_system_window and surface.get('surface_type') != 'CHAT_UI':
            comic_text = comic_text.upper()
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
        # Right-angle caption: fit in the transposed canvas (so the translation
        # can use the caption's *length*), keep the page-space surface for the
        # glyph guard. right_angle_fit_space() inverts exactly the turn
        # paint_fitted_text applies, so the rotated glyphs land back inside the
        # raw footprint.
        right_angle=bool(rotation) and abs(abs(rotation)-90)<1
        page_safe_mask=mask
        def fit_space(current_mask,current_anchor,box=bounds,footprint=None):
            return right_angle_fit_space(current_mask,current_anchor,box,rotation if right_angle else 0,footprint=footprint)
        mask,anchor_center=fit_space(mask,anchor_center,footprint=b)
        # Mask dari footprint teks Inggris sering cuma setebal baris sumber,
        # sehingga terjemahan dipaksa kecil dan mepet satu sisi balon. Untuk
        # balon datar, pakai interior polygon (inset) sebagai area fit.
        bubble_center=None
        if not right_angle and str(surface.get('surface_type') or '').startswith('BUBBLE'):
            poly_src=surface.get('safe_polygon')
            if poly_src:
                local_poly=np.zeros(mask.shape,np.uint8)
                cv2.fillPoly(local_poly,[np.array(poly_src,dtype=np.int32)-[int(bounds[0]),int(bounds[1])]],255)
                # Ambil hanya area yang warnanya sama dengan isi balon. Ini
                # membuang gerigi/taji balon dan garis tepi, jadi teks tidak
                # menabrak border (page-005 bubble-003 balon bertaji).
                region=np.asarray(output)[int(bounds[1]):int(bounds[3]),int(bounds[0]):int(bounds[2])]
                if region.size:
                    inside=region[local_poly>0]
                    if inside.size:
                        fill=np.median(inside.reshape(-1,3),axis=0)
                        distance=np.abs(region.astype(np.int16)-fill.astype(np.int16)).max(axis=2)
                        fill_area=(((distance<=45)&(local_poly>0)).astype(np.uint8))*255
                        fill_area=cv2.morphologyEx(fill_area,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
                        _rows,_cols=np.nonzero(fill_area)
                        _pad=max(12,int(round(.075*min(float(np.ptp(_rows)),float(np.ptp(_cols)))))) if len(_rows) else 12
                        fill_area=cv2.erode(fill_area,np.ones((_pad,_pad),np.uint8))
                        if fill_area.any():
                            # Balon bertaji: isi polygon masih mencakup ujung
                            # gerigi, jadi teks bisa menabrak border. Pakai
                            # persegi terbesar di dalam area isi.
                            if polygon_is_irregular(poly_src):
                                inner=largest_inscribed_rect(fill_area)
                                if inner.any() and int((inner>0).sum()) >= .30*int((fill_area>0).sum()):
                                    fill_area=inner
                        if fill_area.any():
                            # Ambil blob putih yang benar-benar berisi teks:
                            # balon dipisahkan dari background halaman oleh
                            # garis tepinya, jadi komponen terpisah.
                            count,labels,stats,_=cv2.connectedComponentsWithStats(fill_area,8)
                            best_label,best_overlap=0,-1
                            footprint=(mask>0)
                            for lab in range(1,count):
                                overlap=int(((labels==lab)&footprint).sum())
                                if overlap>best_overlap:
                                    best_label,best_overlap=lab,overlap
                            if best_label:
                                fill_area=((labels==best_label).astype(np.uint8))*255
                        if fill_area.any():
                            _r,_c=np.nonzero(fill_area)
                            bubble_center=(float((_c.min()+_c.max())/2.),float((_r.min()+_r.max())/2.))
                        if fill_area.any():
                            fill_count=int((fill_area>0).sum())
                            mask_count=int((mask>0).sum())
                            # Jangan pernah memperbesar area fit melewati bbox
                            # teks asli: itu yang membuat teks menembus garis
                            # balon. Cukup potong tepinya dengan area isi.
                            blob=fill_area if fill_area.shape==mask.shape else cv2.resize(
                                fill_area,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)
                            # Flood fill dari tengah blok teks: berhenti di garis
                            # balon, jadi hasilnya persis interior balon walau
                            # balon menempel background putih halaman.
                            gray_local=cv2.cvtColor(region,cv2.COLOR_RGB2GRAY)
                            light=(gray_local>=115).astype(np.uint8)
                            seed=(int(np.median(np.nonzero(mask)[1])),int(np.median(np.nonzero(mask)[0])))
                            filler=light.copy()
                            ff_mask=np.zeros((filler.shape[0]+2,filler.shape[1]+2),np.uint8)
                            cv2.floodFill(filler,ff_mask,seed,2)
                            filled=(filler==2)
                            filled_count=int(filled.sum())
                            if .8*mask_count <= filled_count <= 6*mask_count:
                                blob=(filled.astype(np.uint8))*255
                            blob_count=int((blob>0).sum())
                            if blob_count >= .40*mask_count:
                                _r,_c=np.nonzero(blob)
                                _pad=max(10,int(round(.06*min(float(np.ptp(_r)),float(np.ptp(_c))))))
                                blob=cv2.erode(blob,np.ones((_pad,_pad),np.uint8))
                                if blob.any():
                                    mask=blob
                                    page_safe_mask=mask
                                    _d=cv2.distanceTransform(blob,cv2.DIST_L2,5)
                                    _cy,_cx=np.unravel_index(int(np.argmax(_d)),_d.shape)
                                    bubble_center=(float(_cx),float(_cy))
        # Blok teks dipusatkan pada pusat lingkaran terbesar di dalam balon.
        # Titik ini tahan terhadap ekor/asimetri balon, berbeda dengan pusat bbox.
        if str(surface.get('surface_type') or '').startswith('BUBBLE') and mask.any():
            # Posisi teks Inggris asli adalah acuan paling stabil: balon pada
            # komik sudah ditata agar teks sumber di tengah. Pusat blob/DT bisa
            # tertarik ekor atau area putih halaman.
            raw_box=b
            anchor_center=(float(raw_box['x']+raw_box['width']/2.)-float(bounds[0]),
                           float(raw_box['y']+raw_box['height']/2.)-float(bounds[1]))
        # Bubble chat adalah UI ponsel, bukan lettering komik. Font komik +
        # outline tebal membuat teks terlihat ditempel (pyscho ch2 page-005
        # bubble-004, page-007 bubble-009). Pakai sans polos tanpa outline;
        # warna tetap mengikuti warna teks sumber.
        # Hanya panel ponsel yang benar-benar kotak yang dianggap chat. Balon
        # gelap/oval sering salah dilabeli CHAT_UI dan berakhir kecil + sans.
        chat_like = (surface.get('surface_type') == 'CHAT_UI'
                     and _shape_score(surface.get('safe_polygon') or []) >= .90
                     and float((surface.get('stats') or {}).get('gray_variance') or 1e9) <= 1500
                     and polygon_is_chat_panel(surface.get('safe_polygon') or [])) or (
            surface.get('surface_type') == 'BUBBLE_COLORED'
            and float(surface.get('stats', {}).get('color_ratio') or 0) >= .75
            and float(surface.get('stats', {}).get('gray_variance') or 1e9) <= 1200
            and _shape_score(surface.get('safe_polygon') or []) >= .70
        )
        if chat_like:
            sans_chat = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
            if sans_chat.is_file():
                selected_font = sans_chat
            stroke = 0
            alignment = 'left'
            # Mask chat harus persis isi gelembungnya. Mask bawaan surface bisa
            # berupa persegi ber-pad, sehingga teks tumpah ke background biru
            # (page-007: 91% piksel teks di luar gelembung hijau). Ambil area
            # yang warnanya sama dengan isi bubble di dalam polygon.
            poly_page = np.array(surface.get('safe_polygon') or [], dtype=np.int32)
            if len(poly_page) >= 3:
                lx, ty = int(bounds[0]), int(bounds[1])
                rx, by = int(bounds[2]), int(bounds[3])
                region = np.asarray(output)[ty:by, lx:rx]
                if region.size:
                    pmask = np.zeros(region.shape[:2], np.uint8)
                    cv2.fillPoly(pmask, [poly_page - [lx, ty]], 255)
                    inside = region[pmask > 0]
                    if inside.size:
                        median_color = np.median(inside.reshape(-1, 3), axis=0)
                        distance = np.abs(region.astype(np.int16) - median_color.astype(np.int16)).max(axis=2)
                        bubble_area = ((distance <= 40) & (pmask > 0)).astype(np.uint8) * 255
                        bubble_area = cv2.morphologyEx(bubble_area, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
                        bubble_area = cv2.erode(bubble_area, np.ones((11, 11), np.uint8))
                        if bubble_area.any():
                            mask = bubble_area
                            page_safe_mask = mask
            left_b, top_b, right_b, bottom_b = bounds
            patch = np.asarray(output)[max(0, top_b):bottom_b, max(0, left_b):right_b]
            if patch.size:
                poly = np.array(surface.get('safe_polygon') or [], dtype=np.int32)
                if len(poly) >= 3:
                    local_poly = np.zeros(patch.shape[:2], np.uint8)
                    cv2.fillPoly(local_poly, [poly - [max(0, left_b), max(0, top_b)]], 255)
                    inside = patch[local_poly > 0]
                else:
                    inside = patch.reshape(-1, patch.shape[2])
                # Chat asli memakai warna huruf sumbernya (putih di bubble hijau).
                # Warna itu dipertahankan selama kontrasnya cukup; kalau tidak,
                # baru dibalik supaya tetap terbaca.
                bubble_shade = float(np.median(inside)) if inside.size else float(np.median(patch))
                if inside.size:
                    median_rgb = np.median(inside.reshape(-1, 3), axis=0)
                    saturation = float(median_rgb.max() - median_rgb.min())
                else:
                    saturation = 0.0
                # Bubble chat berwarna (jenuh) di raw memakai teks putih;
                # bubble netral/terang memakai teks gelap.
                if saturation >= 60:
                    color = (255, 255, 255)
                else:
                    color = (24, 24, 24) if bubble_shade > 150 else (255, 255, 255)
                style['color'] = color
                style['fill_mode'] = 'solid'
            typography = {**typography, 'font_variant': 'regular',
                          'style_source': 'CHAT_UI_SANS', 'style_confidence': 1.0}
        fitted=fit_text_v5(comic_text,selected_font,mask,max(22,args.min_font_size),preferred,stroke,anchor_center=anchor_center,alignment=alignment)
        regular_font=args.font_dialogue or args.font
        if not fitted and str(selected_font)!=str(regular_font):
            fitted=fit_constrained_text(comic_text,selected_font,mask,stroke,anchor_center,alignment)
        if not fitted and not is_system_window and str(selected_font)!=str(regular_font):
            fallback=fit_text_v5(comic_text,regular_font,mask,max(22,args.min_font_size),preferred,stroke,anchor_center=anchor_center,alignment=alignment)
            if fallback:
                fitted=fallback;selected_font=regular_font
                typography={**typography,'font_variant':'regular','style_source':'FIT_FALLBACK','style_confidence':1.0,'font_fallback_reason':'VARIANT_DOES_NOT_FIT'}
        if not fitted and flat:
            stats=surface.get('stats',{})
            # Speed-line artwork is sometimes classified as a white balloon.
            # Retry only within the bounded RAW text neighborhood, preserving
            # other text exclusions, with an outline rather than a white card.
            if stats.get('light_ratio',1)<.8 and stats.get('dark_ratio',0)>.1:
                bounds,mask=source_surface_mask(reference,b,{**surface,'surface_type':'ARTWORK_TEXT'},[g['bbox'] for g in groups if g['id']!=group['id']])
                anchor_center=source_anchor(bounds)
                page_safe_mask=mask
                mask,anchor_center=fit_space(mask,anchor_center,box=bounds,footprint=b)
                stroke=2
                fitted=fit_text_v5(comic_text,selected_font,mask,max(22,args.min_font_size),preferred,stroke,anchor_center=anchor_center)
        if not fitted and not is_system_window:
            # Anchor titik tengah glyph sumber bisa buntu di tepi area aman
            # (dasar gelembung terpotong artwork/label tetangga), sehingga blok
            # teks tidak pernah muat walau mask-nya lapang dan terjemahan lalu
            # dibuang (PRESERVED/UNFIT_SOURCE_RESTORED -> teks Inggris balik).
            # Tengah area aman selalu bisa dipakai, jadi coba di sana dulu
            # sebelum menyerah ke lantai darurat.
            for candidate_font in (selected_font,regular_font):
                if candidate_font is not selected_font and str(candidate_font)==str(selected_font):
                    continue
                centered=fit_mask_center(comic_text,candidate_font,mask,max(22,args.min_font_size),preferred,stroke,alignment)
                if not centered:
                    continue
                fitted=centered
                if str(candidate_font)!=str(selected_font):
                    selected_font=candidate_font
                    typography={**typography,'font_variant':'regular','style_source':'FIT_FALLBACK','style_confidence':1.0,'font_fallback_reason':'VARIANT_DOES_NOT_FIT'}
                break
        if not fitted:
            # User-approved emergency floor for genuinely constrained text.
            # Reuse the same safe mask: never enlarge/ignore balloon boundaries.
            fitted=fit_constrained_text(comic_text,selected_font,mask,stroke,anchor_center,alignment)
        if not fitted:
            # Perspective cards/SFX can receive an over-conservative inferred
            # polygon or collide with a neighbouring detection. The original
            # OCR footprint is always a valid last bounded placement area.
            x,y,w,h=(int(b[k]) for k in ('x','y','width','height'))
            raw_mask=np.full((h,w),255,np.uint8)
            raw_mask=cv2.erode(raw_mask,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
            raw_fitted=fit_constrained_text(comic_text,selected_font,raw_mask,stroke,(w/2,h/2))
            if not raw_fitted and stroke:
                raw_fitted=fit_constrained_text(comic_text,selected_font,raw_mask,0,(w/2,h/2))
                if raw_fitted:
                    stroke=0
            if raw_fitted and not right_angle:
                bounds,mask,fitted=(x,y,x+w,y+h),raw_mask,raw_fitted
                anchor_center=(w/2,h/2)
                fitted['fallback_reason']='RAW_FOOTPRINT_18PX_FLOOR'
                page_safe_mask=mask
            elif raw_fitted and right_angle:
                # Keep the caption's right angle: the rescue fit must run on the
                # transposed footprint, and the guard must switch to the
                # page-space footprint that bounds now describes.
                rescue_mask,rescue_anchor=fit_space(raw_mask,(w/2,h/2),box=(x,y,x+w,y+h),footprint=b)
                rescue=fit_constrained_text(comic_text,selected_font,rescue_mask,stroke,rescue_anchor)
                if rescue:
                    bounds,mask,fitted=(x,y,x+w,y+h),rescue_mask,rescue
                    anchor_center=rescue_anchor
                    page_safe_mask=raw_mask
                    fitted['fallback_reason']='RAW_FOOTPRINT_18PX_FLOOR'
        if not fitted and is_tiny_sfx and stroke:
            fitted=fit_constrained_text(comic_text,selected_font,mask,0,anchor_center)
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
        # An exterior comment re-bases bounds and the surface together, so the
        # page-space guard has to follow it or paint_fitted_text compares a
        # glyph layer against a differently shaped mask. A right-angle caption
        # keeps its transposed fit instead -- re-basing it would un-rotate the
        # caption it was just measured against.
        if not right_angle:
            for candidate_bounds,candidate_mask in exterior_candidates:
                candidate=fit_text_v5(comic_text,selected_font,candidate_mask,max(22,args.min_font_size),preferred,stroke)
                if candidate:
                    bounds,mask,fitted=candidate_bounds,candidate_mask,candidate
                    page_safe_mask=mask
                    placement='automatic-exterior-comment-v1'
                    break
        if str(surface.get('surface_type') or '').startswith('BUBBLE') and not right_angle:
            fitted=recenter_fitted_block(fitted,mask,target=anchor_center)
        elif not right_angle:
            # SFX/kata besar di atas artwork harus tetap di posisi teks aslinya;
            # pusat area aman bisa melenceng ke bawah.
            want=(float(b['x']+b['width']/2)-float(bounds[0]),float(b['y']+b['height']/2)-float(bounds[1]))
            fitted=recenter_fitted_block(fitted,mask,target=want)
        x1,y1,x2,y2=bounds
        ox,oy=fitted['origin']
        local=np.array(output)[y1:y2,x1:x2]
        dark=float(np.median(local))<100
        outline_color=tuple(style.get('outline_color', (0,0,0) if sum(color)>600 else (255,255,255) if not dark else (0,0,0)))
        if dark and max(color)<150 and stroke:
            outline_color=(255,255,255)
        if stroke:
            # Outline tidak boleh menyatu dengan isi huruf: putih-di-atas-putih
            # membuat blok putih tak terbaca (pyscho ch3 page-005 bubble-008/009:
            # huruf putih dengan ring putih 4 px). Kalau balonnya berwarna gelap,
            # tiru raw: halo memakai warna isi balon, bukan hitam pekat.
            text_luminance=float(np.mean(color))
            fill_rgb=np.median(local.reshape(-1,3),axis=0)
            fill_luminance=float(np.mean(fill_rgb))
            saturated=float(fill_rgb.max()-fill_rgb.min()) >= 45
            if text_luminance > 170 and saturated and fill_luminance < 150:
                outline_color=tuple(int(round(v)) for v in fill_rgb)
            elif abs(text_luminance-float(np.mean(outline_color))) < 90:
                outline_color=(0,0,0) if text_luminance > 128 else (255,255,255)
        style.update(rendered_outline_color=outline_color,rendered_outline_width=stroke,actual_font=str(selected_font))
        if fitted.get('layout_balance_reason'):style['layout_balance_reason']=fitted['layout_balance_reason']
        outline='black' if outline_color==(0,0,0) else 'white'
        # Right-angle captions fitted in the transposed canvas still have to be
        # guarded against the *page* surface, which is what page_safe_mask holds.
        rotation_safe_mask=page_safe_mask
        if rotation and surface.get('surface_type')=='TILTED_CARD':
            # Adjacent form fields make the unrotated exclusion mask extremely
            # tight at glyph corners. Permit only a small card-local keyline
            # gutter so the measured RAW angle survives without invading a
            # neighbouring field.
            rotation_gutter=min(24,max(8,round(
                min(mask.shape)*abs(math.sin(math.radians(rotation)))/2
            )))
            rotation_safe_mask=cv2.dilate(
                page_safe_mask,np.ones((rotation_gutter*2+1,rotation_gutter*2+1),np.uint8),iterations=1
            )
        # fitted origin is local to bounds; paint through measured glyph masks.
        painted=paint_fitted_text(
            output,bounds,fitted,style,stroke,
            outline_color,
            rotation=rotation,rotation_center=anchor_center,safe_mask=rotation_safe_mask,
        )
        if not painted and rotation:
            # A translated block is often wider than its RAW source. Preserve
            # the measured RAW angle by reducing size before giving up on the
            # rotation; never clip rotated glyphs against the safe surface.
            for rotated_size in range(fitted['font_size']-1,17,-1):
                rotated_fit=fit_text_v5(
                    comic_text,selected_font,mask,
                    rotated_size,rotated_size,stroke,
                    anchor_center=anchor_center,
                )
                if not rotated_fit:
                    continue
                if paint_fitted_text(
                    output,bounds,rotated_fit,style,stroke,
                    outline_color,
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
                outline_color,
                rotation=rotation,rotation_center=anchor_center,
                safe_mask=np.full(page_safe_mask.shape,255,np.uint8),
            )
            if painted:
                placement+='-tilted-card-bounds'
        if not painted:
            rotation=0.0
            paint_fitted_text(
                output,bounds,fitted,style,stroke,
                outline_color,
            )
        results.append({'id':group['id'],'bbox':{'x':x1,'y':y1,'width':x2-x1,'height':y2-y1},'status':'TYPESET','layout_engine':'v5','placement_strategy':placement,'preferred_font_size':preferred,'font':str(selected_font),'rotation_degrees':rotation,**typography,**style,**{k:fitted[k] for k in ('font_size','wrapped','line_count','overflow_pixels')}})
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    output.save(args.output, optimize=False, compress_level=3)
    Path(args.report).write_text(json.dumps({'layout_engine':'v5','input_dimensions':list(output.size),'output_dimensions':list(output.size),'results':results,'automatic_promotions':promo_decisions},indent=2))
    return 0
