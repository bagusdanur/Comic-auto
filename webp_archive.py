"""Build compact WebP chapter archives while retaining PNG review masters."""
from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

MAX_CONVERT_HEIGHT = 16000
SPLIT_HEIGHT = 12000
WEBP_QUALITY = 100
# method=4: 2,5x lebih cepat dari method=6 dengan selisih ukuran hanya ~1,2%
# (terukur pada halaman nyata 720x15000: 2,10s vs 5,29s; 2,47MB vs 2,44MB).
# method=2 lebih cepat lagi tapi ukuran +41%, jadi tidak dipakai.
WEBP_METHOD = 4


def name_suffix() -> str:
    """Akhiran nama file WebP dari env `WEBP_NAME_SUFFIX`.

    Dipakai untuk membedakan hasil render baru dari file lama yang sudah
    ter-cache di pembaca (mis. `001_001-v2.webp`). Kosong = perilaku lama.
    """
    raw = os.environ.get("WEBP_NAME_SUFFIX", "").strip()
    if not raw:
        return ""
    return raw if raw.startswith(("-", "_")) else "-" + raw


def member_name(stem: str, part: int | None = None) -> str:
    suffix = name_suffix()
    if part is None:
        return f"{stem}{suffix}.webp"
    return f"{stem}{suffix}_part{part:02d}.webp"


def _webp_bytes(image: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="WEBP",
        quality=quality,
        method=WEBP_METHOD,
        exact=True,
    )
    return buffer.getvalue()


def page_webp_members(
    source: Path,
    quality: int = WEBP_QUALITY,
    crop_top: int = 0,
    crop_bottom: int = 0,
) -> list[tuple[str, bytes]]:
    """WebP members for ONE rendered page, named exactly as `archive_webp` does.

    A page taller than MAX_CONVERT_HEIGHT is split into SPLIT_HEIGHT parts so a
    single-page download carries the same member(s) the chapter archive holds for
    that page — `001_001.webp`, or `001_001_part01.webp` + `..._part02.webp`.

    `crop_top` / `crop_bottom` trim that many pixels off the page before anything
    is written. This is how an operator removes a leftover source banner or
    promo strip from the delivered page without touching the review master.
    """
    with Image.open(source) as opened:
        stem = source.stem
        page = crop_page(opened, crop_top, crop_bottom)
        if page.height <= MAX_CONVERT_HEIGHT:
            return [(member_name(stem), _webp_bytes(page, quality))]
        members: list[tuple[str, bytes]] = []
        part = 1
        for y in range(0, page.height, SPLIT_HEIGHT):
            chunk = page.crop((0, y, page.width, min(page.height, y + SPLIT_HEIGHT)))
            members.append((member_name(stem, part), _webp_bytes(chunk, quality)))
            part += 1
        return members


def crop_page(image: Image.Image, crop_top: int, crop_bottom: int) -> Image.Image:
    """Trim `crop_top` px from the top and `crop_bottom` px from the bottom.

    Returns the image unchanged when both trims are zero, and never returns an
    empty page: at least one row survives even if the trims overshoot. PIL's
    `crop` does not clamp its box — it pads anything outside the image — so the
    box is clamped here instead.
    """
    top = max(0, int(crop_top or 0))
    bottom = max(0, int(crop_bottom or 0))
    if not top and not bottom:
        return image
    width, height = image.size
    top = min(top, height - 1)
    bottom_edge = min(height, max(top + 1, height - bottom))
    if top == 0 and bottom_edge == height:
        return image
    return image.crop((0, top, width, bottom_edge))


def archive_webp(
    images: list[Path],
    destination: Path,
    quality: int = WEBP_QUALITY,
    crops: dict[str, tuple[int, int]] | None = None,
) -> int:
    """Atomically archive images as WebP using dashboard-converter settings.

    `crops` maps a source file name to `(crop_top, crop_bottom)` so the archive
    can deliver per-page trimmed output while the review masters stay untouched.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    crops = crops or {}
    count = 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for source in sorted(images, key=lambda item: item.name.casefold()):
                with Image.open(source) as opened:
                    stem = source.stem
                    top, bottom = crops.get(source.name, (0, 0))
                    page = crop_page(opened, top, bottom)
                    if page.height <= MAX_CONVERT_HEIGHT:
                        archive.writestr(member_name(stem), _webp_bytes(page, quality))
                        count += 1
                        continue
                    part = 1
                    for y in range(0, page.height, SPLIT_HEIGHT):
                        chunk = page.crop((0, y, page.width, min(page.height, y + SPLIT_HEIGHT)))
                        archive.writestr(member_name(stem, part), _webp_bytes(chunk, quality))
                        count += 1
                        part += 1
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count
