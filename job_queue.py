#!/usr/bin/env python3
"""Manage persistent translator jobs from the command line."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

from PIL import Image

from queue_db import connect, new_id, row_dict

# Formats the pipeline can decode. Mist (and several CDNs) serve AVIF pages;
# Pillow 12 in this venv decodes AVIF natively, so it must be accepted as input.
SUPPORTED_INPUTS = {".png", ".jpg", ".jpeg", ".webp", ".avif"}
# Formats we can write back. AVIF is decode-only here -- pages are normalised to
# PNG -- so it must never be offered as an output extension.
SUPPORTED_OUTPUTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_IMAGES = 60
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_TOTAL_BYTES = 250 * 1024 * 1024


def safe_name(name: str, index: int) -> str:
    clean = "".join(character if character.isalnum() or character in "._-" else "_" for character in Path(name).name)
    return f"{index + 1:03d}_{clean or 'page.png'}"


def folder_slug(value: str, fallback: str) -> str:
    clean = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return (clean[:80] or fallback)


def collect_inputs(sources: list[Path], destination: Path) -> list[Path]:
    stored: list[Path] = []
    total = 0
    for source in sources:
        source = source.resolve()
        if not source.is_file():
            raise ValueError(f"Input does not exist: {source.name}")
        if source.suffix.casefold() == ".zip":
            with zipfile.ZipFile(source) as archive:
                for info in sorted(archive.infolist(), key=lambda item: item.filename.casefold()):
                    member = Path(info.filename.replace("\\", "/"))
                    if info.is_dir() or member.suffix.casefold() not in SUPPORTED_INPUTS:
                        continue
                    if member.is_absolute() or ".." in member.parts:
                        raise ValueError("Unsafe ZIP path detected")
                    if info.file_size <= 0 or info.file_size > MAX_IMAGE_BYTES:
                        raise ValueError("ZIP contains an empty or oversized image")
                    total += info.file_size
                    if total > MAX_TOTAL_BYTES or len(stored) >= MAX_IMAGES:
                        raise ValueError("Batch exceeds 60 images or 250 MB extracted")
                    target = destination / safe_name(member.name, len(stored))
                    with archive.open(info) as reader, target.open("xb") as writer:
                        shutil.copyfileobj(reader, writer, length=1024 * 1024)
                    stored.append(target)
        elif source.suffix.casefold() in SUPPORTED_INPUTS:
            size = source.stat().st_size
            if size <= 0 or size > MAX_IMAGE_BYTES:
                raise ValueError("Image is empty or exceeds 25 MB")
            total += size
            if total > MAX_TOTAL_BYTES or len(stored) >= MAX_IMAGES:
                raise ValueError("Batch exceeds 60 images or 250 MB")
            target = destination / safe_name(source.name, len(stored))
            shutil.copy2(source, target)
            stored.append(target)
        else:
            raise ValueError("Inputs must be PNG, JPEG, WebP, AVIF, or ZIP")
    if not stored:
        raise ValueError("No supported images were found")
    for path in stored:
        with Image.open(path) as image:
            image.verify()
    return stored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/jobs"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    enqueue = commands.add_parser("enqueue")
    enqueue.add_argument("input", type=Path, nargs="+")
    enqueue.add_argument("--output", type=Path)
    enqueue.add_argument("--provider", default="9router")
    enqueue.add_argument("--env-file", default=".env.9router")
    enqueue.add_argument("--glossary", type=Path)
    enqueue.add_argument("--min-y", type=int, default=0)
    enqueue.add_argument("--title", default="Tanpa Judul")
    enqueue.add_argument("--chapter", default="1")
    listing = commands.add_parser("list")
    listing.add_argument("--limit", type=int, default=20)
    show = commands.add_parser("show")
    show.add_argument("id")
    cancel = commands.add_parser("cancel")
    cancel.add_argument("id")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    db_path = args.db if args.db.is_absolute() else root / args.db
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    database = connect(db_path)

    if args.command == "init":
        print(json.dumps({"database": str(db_path), "status": "READY"}, indent=2))
        return 0
    if args.command == "enqueue":
        job_id = new_id()
        comic_title = " ".join(args.title.strip().split())[:120] or "Tanpa Judul"
        chapter = " ".join(args.chapter.strip().split())[:40] or "1"
        title_slug = folder_slug(comic_title, "tanpa-judul")
        chapter_slug = folder_slug(chapter.removeprefix("Chapter ").removeprefix("chapter "), "1")
        job_root = data_dir / "library" / title_slug / f"chapter-{chapter_slug}" / job_id
        original_dir, output_dir = job_root / "original", job_root / "output"
        for folder in (original_dir, output_dir, job_root / "preview", job_root / "metadata", job_root / "temp"):
            folder.mkdir(parents=True, exist_ok=True)
        try:
            stored_inputs = collect_inputs(args.input, original_dir)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            shutil.rmtree(job_root, ignore_errors=True)
            raise SystemExit(str(error)) from error
        stored_input = stored_inputs[0]
        batch = len(stored_inputs) > 1 or any(path.suffix.casefold() == ".zip" for path in args.input)
        requested_output = args.output.resolve() if args.output else None
        output_suffix = requested_output.suffix.casefold() if requested_output else stored_input.suffix.casefold()
        if batch:
            if requested_output and output_suffix != ".zip":
                raise SystemExit("Batch output extension must be ZIP")
            stored_output = output_dir / "chapter-translated.zip"
        else:
            if output_suffix not in SUPPORTED_OUTPUTS:
                raise SystemExit("Output extension must be PNG, JPEG, or WebP")
            stored_output = output_dir / f"{stored_input.stem}{output_suffix}"
        options = {
            "provider": args.provider, "env_file": args.env_file,
            "glossary": str(args.glossary.resolve()) if args.glossary else None,
            "min_y": args.min_y, "batch": batch,
            "input_files": [path.name for path in stored_inputs],
            "comic_title": comic_title, "chapter": chapter,
            "title_slug": title_slug,
        }
        now = time.time()
        database.execute(
            "INSERT INTO jobs(id,status,input_path,output_path,requested_output_path,options_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (job_id, "QUEUED", str(stored_input), str(stored_output),
             str(requested_output) if requested_output else None, json.dumps(options), now, now),
        )
        database.commit()
        print(json.dumps(row_dict(database.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()), indent=2))
        return 0
    if args.command == "list":
        rows = database.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (args.limit,)).fetchall()
        print(json.dumps([row_dict(row) for row in rows], indent=2))
        return 0
    if args.command == "show":
        row = row_dict(database.execute("SELECT * FROM jobs WHERE id=?", (args.id,)).fetchone())
        if not row:
            raise SystemExit("Job not found")
        print(json.dumps(row, indent=2))
        return 0
    if args.command == "cancel":
        now = time.time()
        updated = database.execute(
            "UPDATE jobs SET status='CANCELLED',"
            "stage=CASE WHEN status='PROCESSING' THEN 'CANCEL_REQUESTED' ELSE 'CANCELLED' END,"
            "updated_at=?,completed_at=? "
            "WHERE id=? AND status IN ('QUEUED','WAITING_RESOURCE','PROCESSING')",
            (now, now, args.id),
        )
        database.commit()
        if updated.rowcount != 1:
            raise SystemExit("Job not found or already processing/finished")
        print(json.dumps({"id": args.id, "status": "CANCELLED"}))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
