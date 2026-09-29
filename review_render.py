#!/usr/bin/env python3
"""Generate lightweight previews and safely re-render one translated region."""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from webp_archive import archive_webp, page_webp_members

from typeset_test import fit_text


# Floor for the editor's shrink-to-fit. Mirrors the production typesetter's
# `fit_minimum` (typeset_test.py) so a hand-edited bubble cannot end up with
# smaller text than the pipeline itself would ever emit. `MINIMUM_SCALE_RATIO`
# is the same 0.52 default as `--min-scale-ratio`.
MINIMUM_FONT_SIZE = 14
MINIMUM_SCALE_RATIO = 0.52


def bounded_box(raw: dict, image: Image.Image) -> tuple[int, int, int, int]:
    x = max(0, int(raw["x"]))
    y = max(0, int(raw["y"]))
    width = max(1, min(int(raw["width"]), image.width - x))
    height = max(1, min(int(raw["height"]), image.height - y))
    if x >= image.width or y >= image.height:
        raise ValueError("Region is outside the image")
    return x, y, width, height


def preview(source: Path, destination: Path, max_width: int) -> None:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
        width = min(max_width, image.width)
        height = max(1, round(image.height * width / image.width))
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.save(destination, "JPEG", quality=82, optimize=True)


def render(args: argparse.Namespace) -> None:
    bbox = json.loads(args.bbox)
    original_bbox = json.loads(args.original_bbox)
    color = tuple(json.loads(args.color))
    stroke_color = tuple(json.loads(args.stroke_color))
    with Image.open(args.current) as opened:
        output = opened.convert("RGB")
    with Image.open(args.cleaned) as opened:
        cleaned = opened.convert("RGB")
    if output.size != cleaned.size:
        raise ValueError("Cleaned and output dimensions differ")

    # Restore both the original and edited boxes from the clean pipeline image.
    for raw in (original_bbox, bbox):
        x, y, width, height = bounded_box(raw, output)
        output.paste(cleaned.crop((x, y, x + width, y + height)), (x, y))

    x, y, width, height = bounded_box(bbox, output)
    surface = np.asarray(cleaned.crop((x, y, x + width, y + height)))
    if surface.size:
        gray = np.mean(surface, axis=2)
        if float(np.mean(gray >= 220)) >= 0.60:
            color = (15, 15, 15)
            stroke_color = color
            args.stroke_width = 0
    margin = 3
    inner_width = max(1, width - margin * 2)
    inner_height = max(1, height - margin * 2)
    draw = ImageDraw.Draw(output)
    # Shrink to fit instead of refusing. The production typesetter already lets a
    # translation borrow a smaller face (`fit_minimum`) when the preferred size
    # does not fit, but this editor asked `fit_text` for exactly one size
    # (minimum == maximum), so any edit that did not fit at the chosen size was
    # rejected — the edit saved as status EDITED and the page never re-rendered.
    fit_minimum = max(MINIMUM_FONT_SIZE, round(args.font_size * MINIMUM_SCALE_RATIO))
    fitted = fit_text(
        draw, args.translation.upper(), args.font, inner_width, inner_height,
        fit_minimum, args.font_size, args.line_spacing, args.stroke_width,
    )
    if not fitted:
        raise ValueError(
            f"Teks tidak muat di kotak {width}x{height} bahkan pada font "
            f"{fit_minimum}px. Perbesar kotak (Width/Height) atau pendekkan "
            "terjemahannya."
        )
    text_bbox = fitted["text_bbox"]
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    if args.alignment == "left":
        text_x = x + margin - text_bbox[0]
    elif args.alignment == "right":
        text_x = x + width - margin - text_width - text_bbox[0]
    else:
        text_x = x + (width - text_width) / 2 - text_bbox[0]
    text_y = y + (height - text_height) / 2 - text_bbox[1]
    draw.multiline_text(
        (text_x, text_y), fitted["wrapped"], font=fitted["font"], fill=color,
        spacing=fitted["spacing"], align=args.alignment,
        stroke_width=args.stroke_width, stroke_fill=stroke_color,
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    output.save(temporary, "PNG", optimize=True)
    os.replace(temporary, args.output)


def archive(source: Path, destination: Path) -> None:
    archive_webp(list(source.glob("*.png")), destination)


def page_file(source: Path, directory: Path) -> Path:
    """Write ONE page into `directory` and return the file that was written.

    The extension is decided here because only this side knows whether the page
    is taller than the archive split: a short page becomes `<stem>.webp`, a tall
    one becomes `<stem>.zip` holding the `_partNN.webp` members the chapter ZIP
    uses. The caller receives the concrete path so it never has to guess.
    """
    members = page_webp_members(source)
    directory.mkdir(parents=True, exist_ok=True)
    if len(members) == 1:
        target = directory / f"{source.stem}.webp"
        target.write_bytes(members[0][1])
        return target
    target = directory / f"{source.stem}.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members:
            archive.writestr(name, payload)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    make_preview = commands.add_parser("preview")
    make_preview.add_argument("source", type=Path)
    make_preview.add_argument("destination", type=Path)
    make_preview.add_argument("--max-width", type=int, default=420)
    rerender = commands.add_parser("render")
    rerender.add_argument("cleaned", type=Path)
    rerender.add_argument("current", type=Path)
    rerender.add_argument("output", type=Path)
    rerender.add_argument("--bbox", required=True)
    rerender.add_argument("--original-bbox", required=True)
    rerender.add_argument("--translation", required=True)
    rerender.add_argument("--font", default="assets/fonts/wildwords/CC Wild Words Roman.ttf")
    rerender.add_argument("--font-size", type=int, required=True)
    rerender.add_argument("--line-spacing", type=float, default=0.12)
    rerender.add_argument("--alignment", choices=("left", "center", "right"), default="center")
    rerender.add_argument("--color", default="[15,15,15]")
    rerender.add_argument("--stroke-width", type=int, default=0)
    rerender.add_argument("--stroke-color", default="[15,15,15]")
    make_archive = commands.add_parser("archive")
    make_archive.add_argument("source", type=Path)
    make_archive.add_argument("destination", type=Path)
    make_page = commands.add_parser("page")
    make_page.add_argument("source", type=Path)
    make_page.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        preview(args.source, args.destination, args.max_width)
    elif args.command == "archive":
        archive(args.source, args.destination)
    elif args.command == "page":
        print(page_file(args.source, args.directory))
    else:
        render(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
