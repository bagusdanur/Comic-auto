#!/usr/bin/env python3
"""Learn title-specific terminology and style from one paired translated chapter."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from PIL import Image
import psutil

from job_queue import collect_inputs
from queue_db import connect
from translate_text_test import group_dialogue, load_env
from translation_providers import NineRouterProvider, TranslationError, parse_json_object


def update(database, job_id: str, stage: str, progress: float, **values) -> None:
    now = time.time()
    assignments = ["stage=?", "progress=?", "updated_at=?", "heartbeat_at=?"]
    parameters: list[object] = [stage, progress, now, now]
    for key, value in values.items():
        assignments.append(f"{key}=?")
        parameters.append(value)
    parameters.append(job_id)
    database.execute(f"UPDATE memory_jobs SET {','.join(assignments)} WHERE id=?", parameters)
    database.commit()


def normalized_box(box: dict, width: int, height: int) -> tuple[float, float, float, float]:
    return (
        (box["x"] + box["width"] / 2) / max(width, 1),
        (box["y"] + box["height"] / 2) / max(height, 1),
        box["width"] / max(width, 1), box["height"] / max(height, 1),
    )


def match_groups(raw: dict, translated: dict) -> list[dict]:
    source = group_dialogue(raw.get("regions", []), 0)
    target = group_dialogue(translated.get("regions", []), 0)
    candidates: list[tuple[float, int, int]] = []
    for source_index, left in enumerate(source):
        sx, sy, sw, sh = normalized_box(left["bbox"], raw["width"], raw["height"])
        for target_index, right in enumerate(target):
            tx, ty, tw, th = normalized_box(right["bbox"], translated["width"], translated["height"])
            center = math.hypot(sx - tx, sy - ty)
            size = abs(math.log(max(sw, .001) / max(tw, .001))) + abs(math.log(max(sh, .001) / max(th, .001)))
            score = center * 5.5 + size * .15
            candidates.append((score, source_index, target_index))
    used_source: set[int] = set()
    used_target: set[int] = set()
    pairs: list[dict] = []
    for score, source_index, target_index in sorted(candidates):
        if source_index in used_source or target_index in used_target or score > 1.35:
            continue
        used_source.add(source_index)
        used_target.add(target_index)
        confidence = max(0.0, min(1.0, 1.0 - score / 1.35))
        pairs.append({
            "source": source[source_index]["text"], "translation": target[target_index]["text"],
            "confidence": round(confidence, 3), "source_bbox": source[source_index]["bbox"],
            "translated_bbox": target[target_index]["bbox"],
        })
    return sorted(pairs, key=lambda item: item["source_bbox"]["y"])


async def analyze_with_ai(comic_title: str, chapter: str, pairs: list[dict]) -> dict:
    safe_pairs = [item for item in pairs if item["confidence"] >= 0.45][:240]
    prompt = f"""Analyze a professional English-to-Indonesian comic translation reference.
Comic: {comic_title}; chapter: {chapter}.
Return JSON only with this schema:
{{"terminology":[{{"source":"","target":"","category":"character|skill|organization|rank|item|term","locked":true,"confidence":0.0}}],
"character_voices":[{{"character":"","voice":"","examples":[""]}}],
"style_profile":{{"register":"","pronouns":"","dialogue_rules":[""],"narration_rules":[""],"thought_rules":[""],"shout_rules":[""]}},
"translation_examples":[{{"source":"","translation":"","reason":""}}],"summary":""}}.
Preserve canonical named skills and proper names in English when the Indonesian reference does so.
Do not turn ordinary words into terminology. Use only evidence from the supplied pairs.
Pairs:
{json.dumps(safe_pairs, ensure_ascii=False)}"""
    provider = NineRouterProvider()
    result = None
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            raw = await provider._request(prompt, {})
            if not isinstance(raw, str) or not raw.strip():
                raise TranslationError("9router mengembalikan respons kosong")
            candidate = parse_json_object(raw)
            if not isinstance(candidate, dict):
                raise TranslationError("respons analisis bukan objek JSON")
            result = candidate
            break
        except (KeyError, TypeError, ValueError, TranslationError) as error:
            last_error = error
            if attempt < 3:
                await asyncio.sleep(attempt * 2)
    if result is None:
        raise TranslationError(f"Analisis AI gagal setelah 3 percobaan: {last_error}")
    result["pair_count"] = len(pairs)
    result["high_confidence_pair_count"] = sum(item["confidence"] >= 0.65 for item in pairs)
    result["chapter"] = chapter
    return result


def process_job(database, job: dict, root: Path) -> None:
    job_id = job["id"]
    work = Path(job["work_path"])
    raw_dir, translated_dir = work / "raw", work / "translated"
    for folder in (raw_dir, translated_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
    try:
        update(database, job_id, "EXTRACTING", .04)
        raw_pages = collect_inputs([Path(job["raw_zip_path"])], raw_dir)
        translated_pages = collect_inputs([Path(job["translated_zip_path"])], translated_dir)
        if len(raw_pages) != len(translated_pages):
            raise ValueError(f"Jumlah halaman berbeda: raw {len(raw_pages)}, Indonesia {len(translated_pages)}")
        pairs: list[dict] = []
        for index, (raw_page, translated_page) in enumerate(zip(raw_pages, translated_pages)):
            with Image.open(raw_page) as raw_image, Image.open(translated_page) as translated_image:
                ratio = (raw_image.width / raw_image.height) / (translated_image.width / translated_image.height)
                if not 0.82 <= ratio <= 1.22:
                    raise ValueError(f"Rasio halaman {index + 1} tidak cocok")
            page_dir = work / "pages" / f"{index + 1:03d}"
            raw_ocr, translated_ocr = page_dir / "raw", page_dir / "translated"
            raw_ocr.mkdir(parents=True, exist_ok=True)
            translated_ocr.mkdir(parents=True, exist_ok=True)
            update(database, job_id, f"OCR_RAW [{index + 1}/{len(raw_pages)}]", .08 + index / len(raw_pages) * .36)
            subprocess.run([sys.executable, str(root / "translate_test.py"), str(raw_page), "--output-dir", str(raw_ocr)], cwd=root, check=True, stdout=subprocess.DEVNULL)
            update(database, job_id, f"OCR_TRANSLATED [{index + 1}/{len(raw_pages)}]", .44 + index / len(raw_pages) * .36)
            subprocess.run([sys.executable, str(root / "translate_test.py"), str(translated_page), "--output-dir", str(translated_ocr)], cwd=root, check=True, stdout=subprocess.DEVNULL)
            raw_data = json.loads((raw_ocr / "ocr-result.json").read_text(encoding="utf-8"))
            translated_data = json.loads((translated_ocr / "ocr-result.json").read_text(encoding="utf-8"))
            for item in match_groups(raw_data, translated_data):
                item["page"] = index + 1
                pairs.append(item)
        (work / "matched-pairs.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
        update(database, job_id, "ANALYZING", .86)
        comic = database.execute("SELECT title FROM comics WHERE id=?", (job["comic_id"],)).fetchone()
        proposal = asyncio.run(analyze_with_ai(comic["title"], job["chapter_number"], pairs))
        update(database, job_id, "WAITING_APPROVAL", 1.0, status="WAITING_APPROVAL", proposal_json=json.dumps(proposal, ensure_ascii=False), completed_at=time.time())
    except Exception as error:
        update(database, job_id, "FAILED", 1.0, status="FAILED", error=str(error), completed_at=time.time())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    load_env(root / ".env.9router")
    database = connect(args.db if args.db.is_absolute() else root / args.db)
    now = time.time()
    database.execute(
        "UPDATE memory_jobs SET status='QUEUED',stage='RECOVERED_AFTER_RESTART',progress=0,"
        "updated_at=?,started_at=NULL,heartbeat_at=NULL,error=NULL WHERE status='PROCESSING'",
        (now,),
    )
    database.commit()
    while True:
        now = time.time()
        database.execute("BEGIN IMMEDIATE")
        row = database.execute("SELECT * FROM memory_jobs WHERE status='QUEUED' ORDER BY created_at LIMIT 1").fetchone()
        normal_busy = database.execute("SELECT 1 FROM jobs WHERE status='PROCESSING' LIMIT 1").fetchone()
        enough_ram = psutil.virtual_memory().available >= 2_000 * 1024 * 1024
        if row and not normal_busy and enough_ram and not database.execute("SELECT 1 FROM memory_jobs WHERE status='PROCESSING' LIMIT 1").fetchone():
            database.execute("UPDATE memory_jobs SET status='PROCESSING',stage='VALIDATING',started_at=?,updated_at=? WHERE id=?", (now, now, row["id"]))
            database.commit()
            process_job(database, dict(row), root)
        else:
            if row and (normal_busy or not enough_ram):
                database.execute("UPDATE memory_jobs SET stage='WAITING_RESOURCE',updated_at=? WHERE id=?", (now, row["id"]))
            database.commit()
            if args.once:
                return 0
            time.sleep(5)
        if args.once:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
