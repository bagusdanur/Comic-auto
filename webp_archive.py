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
WEBP_METHOD = 6


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


def page_webp_members(source: Path, quality: int = WEBP_QUALITY) -> list[tuple[str, bytes]]:
    """WebP members for ONE rendered page, named exactly as `archive_webp` does.

    A page taller than MAX_CONVERT_HEIGHT is split into SPLIT_HEIGHT parts so a
    single-page download carries the same member(s) the chapter archive holds for
    that page — `001_001.webp`, or `001_001_part01.webp` + `..._part02.webp`.
    """
    with Image.open(source) as opened:
        stem = source.stem
        if opened.height <= MAX_CONVERT_HEIGHT:
            return [(f"{stem}.webp", _webp_bytes(opened, quality))]
        members: list[tuple[str, bytes]] = []
        part = 1
        for y in range(0, opened.height, SPLIT_HEIGHT):
            chunk = opened.crop((0, y, opened.width, min(opened.height, y + SPLIT_HEIGHT)))
            members.append((f"{stem}_part{part:02d}.webp", _webp_bytes(chunk, quality)))
            part += 1
        return members


def archive_webp(images: list[Path], destination: Path, quality: int = WEBP_QUALITY) -> int:
    """Atomically archive images as WebP using dashboard-converter settings."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for source in sorted(images, key=lambda item: item.name.casefold()):
                with Image.open(source) as opened:
                    stem = source.stem
                    if opened.height <= MAX_CONVERT_HEIGHT:
                        archive.writestr(f"{stem}.webp", _webp_bytes(opened, quality))
                        count += 1
                        continue
                    part = 1
                    for y in range(0, opened.height, SPLIT_HEIGHT):
                        chunk = opened.crop((0, y, opened.width, min(opened.height, y + SPLIT_HEIGHT)))
                        archive.writestr(f"{stem}_part{part:02d}.webp", _webp_bytes(chunk, quality))
                        count += 1
                        part += 1
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count
