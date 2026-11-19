"""Conservative deterministic removal for verified plain white OCR footprints."""
import cv2
import numpy as np


def verified_white_fill(original, source_regions, surface):
    if surface.get('surface_type') != 'BUBBLE_FLAT' or not source_regions:
        return None
    from footprint_masks import source_footprint_batches
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    for batch in source_footprint_batches(original.shape, source_regions):
        mask |= batch
    ys, xs = np.where(mask > 0)
    if not len(xs): return None
    pad = 8
    x0, x1 = max(0, int(xs.min())-pad), min(original.shape[1], int(xs.max())+pad+1)
    y0, y1 = max(0, int(ys.min())-pad), min(original.shape[0], int(ys.max())+pad+1)
    crop = original[y0:y1, x0:x1]; local = mask[y0:y1, x0:x1]
    ring = (cv2.dilate(local, np.ones((13, 13), np.uint8)) > 0) & (local == 0)
    samples = crop[ring]
    if len(samples) < 100: return None
    white = (samples.min(axis=1) >= 248) & ((samples.max(axis=1).astype(int)-samples.min(axis=1)) <= 6)
    if float(white.mean()) < .995: return None
    fill = np.median(samples[white], axis=0).round().astype(np.uint8)
    selected = crop[local > 0]
    chroma = selected.max(axis=1).astype(int)-selected.min(axis=1)
    if float((chroma > 12).mean()) > .002: return None
    if float((selected.min(axis=1) >= 240).mean()) < .5: return None
    ink = ((crop.min(axis=2) < 180) & (local > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    components = [row for row in stats[1:] if row[cv2.CC_STAT_AREA] >= 5]
    if len(components) < 3: return None
    max_height = max(int(r['bbox']['height']) for r in source_regions)
    for row in components:
        w, h = row[cv2.CC_STAT_WIDTH], row[cv2.CC_STAT_HEIGHT]
        if h > max_height * 1.3 or w / max(1, h) > 10: return None
    return mask, fill.tolist(), {'ring_white_fraction': round(float(white.mean()), 5), 'glyph_components': len(components)}


def verified_uniform_fill(original, source_regions, surface):
    """Fill solid warna interior balon yang SERAGAM (tidak harus putih murni).

    `verified_white_fill` menolak balon off-white/bernoise JPEG/berwarna lalu
    engine jatuh ke LaMa inpaint yang meninggalkan patch blur/abu. Jalur ini
    tetap deterministik & crisp: selama cincin sekeliling teks seragam (flat),
    isi glyph dengan warna median cincin. Guard menolak artwork bertekstur.
    """
    if surface.get('surface_type') not in ('BUBBLE_FLAT', 'BUBBLE_COLORED') or not source_regions:
        return None
    from footprint_masks import source_footprint_batches
    mask = np.zeros(original.shape[:2], dtype=np.uint8)
    for batch in source_footprint_batches(original.shape, source_regions):
        mask |= batch
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    pad = 8
    x0, x1 = max(0, int(xs.min()) - pad), min(original.shape[1], int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(original.shape[0], int(ys.max()) + pad + 1)
    crop = original[y0:y1, x0:x1]; local = mask[y0:y1, x0:x1]
    ring = (cv2.dilate(local, np.ones((13, 13), np.uint8)) > 0) & (local == 0)
    samples = crop[ring]
    if len(samples) < 100:
        return None
    fill = np.median(samples, axis=0)
    dev = np.abs(samples.astype(int) - fill.astype(int)).max(axis=1)
    uniform_fraction = float((dev <= 18).mean())
    # Cincin harus warna solid seragam. uniform_fraction robust terhadap outlier
    # (sedikit piksel tepi/garis balon tidak menaikkan std sebanyak fraksi).
    # Artwork bertekstur punya fraksi seragam rendah -> tetap ditolak.
    if uniform_fraction < 0.97:
        return None
    fill = fill.round().astype(np.uint8)
    dist2d = np.abs(crop.astype(int) - fill.astype(int)).max(axis=2)
    selected_dist = dist2d[local > 0]
    # Area teks harus didominasi latar (huruf minoritas); kalau tidak, itu artwork.
    if float((selected_dist > 40).mean()) > 0.6:
        return None
    ink = ((dist2d > 40) & (local > 0)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    components = [row for row in stats[1:] if row[cv2.CC_STAT_AREA] >= 5]
    if len(components) < 2:
        return None
    max_height = max(int(r['bbox']['height']) for r in source_regions)
    for row in components:
        w, h = row[cv2.CC_STAT_WIDTH], row[cv2.CC_STAT_HEIGHT]
        if h > max_height * 1.4 or w / max(1, h) > 12:
            return None
    return mask, fill.tolist(), {
        'ring_uniform_fraction': round(uniform_fraction, 5),
        'fill_color': fill.tolist(), 'glyph_components': len(components),
    }
