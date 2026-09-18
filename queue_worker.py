#!/usr/bin/env python3
"""Single-concurrency persistent worker for translator jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import psutil
from PIL import Image

from brand_postprocess import apply_branding
from branding_coordinates import project_region
from queue_db import claim_next, connect
from webp_archive import archive_webp


def available_ram_mb() -> int:
    return int(psutil.virtual_memory().available / 1024 / 1024)


def load_one_minute() -> float:
    return float(psutil.getloadavg()[0])


def retryable_page_failure(error: str, returncode: int) -> bool:
    """Failures caused by temporary machine/provider pressure should self-heal."""
    message = error.casefold()
    markers = (
        "resource_guard", "available ram", "timed out", "timeout",
        "connection", "bad gateway", "service unavailable", "too many requests",
        "http 429", "http 500", "http 502", "http 503", "http 504",
        "translation failed after", "retain '",
    )
    return returncode in (2, 3, 75) or any(marker in message for marker in markers)


def setting(database, key: str, default: str) -> str:
    row = database.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def persist_results(database, job: dict, pipeline_dir: Path, output_path: Path, filename: str, final_status: str) -> None:
    ocr = json.loads((pipeline_dir / "ocr-result.json").read_text(encoding="utf-8"))
    translations = json.loads((pipeline_dir / "translations.json").read_text(encoding="utf-8"))
    typeset = json.loads((pipeline_dir / "typeset-report.json").read_text(encoding="utf-8"))
    typeset_status = {item["id"]: item for item in typeset["results"]}
    branding_path=pipeline_dir/'branding-report.json'
    branding=json.loads(branding_path.read_text()) if branding_path.exists() else {}
    image_id = str(uuid.uuid4())
    with Image.open(output_path) as final_image:
        final_width, final_height = final_image.size
    database.execute(
        "INSERT INTO images(id,job_id,filename,width,height,status,output_path,metadata_path,region_count) VALUES(?,?,?,?,?,?,?,?,?)",
        (image_id, job["id"], filename, final_width, final_height,
         final_status, str(output_path), str(pipeline_dir), len(translations["groups"])),
    )
    for group in translations["groups"]:
        item = typeset_status[group["id"]]
        if item.get('status')=='REMOVED_PROMO':continue
        projected=project_region(item,group['bbox'],branding)
        if projected is None:continue
        display_box,item=projected
        database.execute(
            "INSERT INTO regions(id,image_id,source_text,translation,bbox_json,status,metadata_json) VALUES(?,?,?,?,?,?,?)",
            (str(uuid.uuid4()), image_id, group["text"], group["translation"],
             json.dumps(display_box), item["status"], json.dumps(item)),
        )
    database.execute('UPDATE images SET region_count=(SELECT count(*) FROM regions WHERE image_id=?) WHERE id=?',(image_id,image_id))


def process_job(database, job: dict, root: Path, *, resume: bool = False) -> None:
    options = job["options"]
    job_root = Path(job["input_path"]).parent.parent
    filenames = options.get("input_files") or [Path(job["input_path"]).name]
    original_names = options.get("original_names") or []
    input_paths=[job_root/'original'/name for name in filenames]
    from priority_seams import PRIORITY_SLUG,prepare_inputs
    if options.get('title_slug')==PRIORITY_SLUG and options.get('pipeline_engine')=='v5':
        input_paths=prepare_inputs(input_paths,job_root/'prepared-seams-v1')
    outputs: list[Path] = []
    review_count = 0
    if not resume:
        database.execute("DELETE FROM images WHERE job_id=?", (job["id"],))
    database.commit()
    for index, filename in enumerate(filenames):
        input_path = input_paths[index]
        batch = bool(options.get("batch"))
        output_path = job_root / "output" / f"{Path(filename).stem}.png" if batch else Path(job["output_path"])
        pipeline_dir = job_root / "metadata" / f"page-{index + 1:03d}"
        cached_pipeline=None
        if resume and (pipeline_dir/'pipeline-status.json').exists():
            saved=json.loads((pipeline_dir/'pipeline-status.json').read_text())
            if saved.get('status')=='COMPLETED' and Path(saved.get('input','')).resolve()==input_path.resolve():
                if all((pipeline_dir/name).exists() for name in ('typeset.png','typeset-report.json','translations.json','ocr-result.json')):
                    cached_pipeline=saved
        if resume:
            existing=database.execute('SELECT status,output_path FROM images WHERE job_id=? AND metadata_path=?',(job['id'],str(pipeline_dir))).fetchone()
            if existing:
                if cached_pipeline is None or existing['status']!='COMPLETED' or Path(existing['output_path'])!=output_path:
                    raise RuntimeError('Cannot safely resume inconsistent page: '+pipeline_dir.name)
                with Image.open(output_path) as image:image.verify()
                outputs.append(output_path)
                review_count+=len(cached_pipeline.get('review_ids',[]))
                print(pipeline_dir.name,'REUSED COMPLETED',flush=True)
                continue
        if cached_pipeline is not None:
            shutil.copy2(pipeline_dir/'typeset.png',output_path)
            print(pipeline_dir.name,'RESUME BRANDING',flush=True)
        command = [
            sys.executable, str(root / "translate.py"), str(input_path), str(output_path),
            "--pipeline-engine", options.get("pipeline_engine") or os.getenv("PIPELINE_ENGINE", "legacy"),
            "--provider", options.get("provider", "9router"),
            "--env-file", options.get("env_file", ".env.9router"),
            "--min-y", str(options.get("min_y", 0)), "--job-dir", str(pipeline_dir),
            "--memory", str(root / "data" / "memory" / f"{options.get('title_slug') or 'legacy'}.json"),
            "--quality-v2", options.get("quality_v2") or os.getenv("QUALITY_PIPELINE_V2", "on"),
            "--hybrid-mode", options.get("hybrid_mode") or os.getenv("HYBRID_PIPELINE", "off"),
            "--lama-model", options.get("lama_model") or os.getenv("LAMA_MODEL_PATH", "models/lama.onnx"),
            "--min-font-size", str(options.get("min_font_size", 22)),
            "--min-scale-ratio", str(options.get("min_scale_ratio", 0.52)),
            "--layout-engine", options.get("layout_engine") or os.getenv("TYPESET_LAYOUT_ENGINE", "legacy"),
        ]
        if options.get("glossary"):
            command += ["--glossary", options["glossary"]]
        # Isolate each page pipeline so dashboard Stop can terminate OCR and
        # every active cleanup/typeset child without leaving orphan processes.
        # Cached page already completed OCR/translation/render; retain the same
        # cancellation/result handling while resuming only its branding step.
        run_command=command if cached_pipeline is None else [sys.executable,'-c','pass']
        child_env = os.environ.copy()
        configured_model = setting(database, "ninerouter_model", "").strip()
        configured_fallbacks = setting(database, "ninerouter_fallback_models", "").strip()
        if configured_model:
            child_env["NINEROUTER_MODEL"] = configured_model
        if configured_fallbacks:
            child_env["NINEROUTER_FALLBACK_MODELS"] = configured_fallbacks
        process = subprocess.Popen(run_command, cwd=root, start_new_session=True, env=child_env)
        cancelled = False
        while process.poll() is None:
            pipeline_status = pipeline_dir / "pipeline-status.json"
            stage = "PROCESSING"
            if pipeline_status.exists():
                try:
                    stage = json.loads(pipeline_status.read_text(encoding="utf-8")).get("stage", stage)
                except (OSError, json.JSONDecodeError):
                    pass
            now = time.time()
            progress = index / len(filenames)
            database.execute(
                "UPDATE jobs SET stage=?,progress=?,updated_at=?,heartbeat_at=? WHERE id=? AND status='PROCESSING'",
                (f"{stage} [{index + 1}/{len(filenames)}]", progress, now, now, job["id"]),
            )
            database.commit()
            current = database.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
            if current and current["status"] == "CANCELLED":
                cancelled = True
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=8)
                except ProcessLookupError:
                    pass
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                break
            time.sleep(2)
        # Cover a cancel request arriving after the child exited but before the
        # worker started collecting its result.
        current = database.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()
        if current and current["status"] == "CANCELLED":
            cancelled = True
        now = time.time()
        if cancelled:
            database.execute(
                "UPDATE jobs SET status='CANCELLED',stage='CANCELLED',updated_at=?,completed_at=?,heartbeat_at=NULL "
                "WHERE id=?",
                (now, now, job["id"]),
            )
            database.commit()
            return
        if process.returncode != 0:
            error = f"Pipeline failed on {filename}"
            status_file = pipeline_dir / "pipeline-status.json"
            if status_file.exists():
                error = json.loads(status_file.read_text(encoding="utf-8")).get("error", error)
            previous = re.search(r"\[auto-retry (\d+)/3\]", str(job.get("error") or ""))
            retry_number = (int(previous.group(1)) + 1) if previous else 1
            if retry_number <= 3 and retryable_page_failure(error, process.returncode):
                retry_error = f"[auto-retry {retry_number}/3] {error}"
                database.execute(
                    "UPDATE jobs SET status='QUEUED',stage=?,error=?,updated_at=?,completed_at=NULL,heartbeat_at=NULL WHERE id=?",
                    (f"AUTO_RETRY [{index + 1}/{len(filenames)}]", retry_error, now, job["id"]),
                )
                database.commit()
                print(f"{pipeline_dir.name} transient failure; scheduled {retry_number}/3", flush=True)
                return
            database.execute(
                "UPDATE jobs SET status='FAILED',stage='FAILED',error=?,updated_at=?,completed_at=? WHERE id=?",
                (error, now, now, job["id"]),
            )
            database.commit()
            return
        pipeline = json.loads((pipeline_dir / "pipeline-status.json").read_text(encoding="utf-8"))
        review_count += len(pipeline.get("review_ids", []))
        if batch and bool(options.get("branding_enabled", True)):
            branding_report = apply_branding(
                output_path,
                pipeline_dir / "ocr-result.json",
                root / "assets" / "branding" / "watermark.png",
                root / "assets" / "branding" / "watermark-text.png",
                first_page=index == 0,
                banner_path=(root / "assets" / "branding" / "banner.png")
                if index == len(filenames) - 1 else None,
            )
            (pipeline_dir / "branding-report.json").write_text(
                json.dumps(branding_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        elif batch:
            (pipeline_dir / "branding-report.json").write_text(
                json.dumps({"enabled": False, "reason": "DISABLED_BY_DEFAULT"}, indent=2),
                encoding="utf-8",
            )
        display_name = original_names[index] if len(original_names) == len(filenames) else filename
        persist_results(database, job, pipeline_dir, output_path, display_name, pipeline["status"])
        outputs.append(output_path)
        database.commit()

    final_status = "REVIEW" if review_count else "COMPLETED"
    if options.get("batch"):
        archive_path = Path(job["output_path"])
        archive_webp(outputs, archive_path)
    requested = job.get("requested_output_path")
    if requested:
        Path(requested).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(job["output_path"], requested)
    now = time.time()
    database.execute(
        "UPDATE jobs SET status=?,stage='DONE',progress=1,review_count=?,updated_at=?,completed_at=? WHERE id=?",
        (final_status, review_count, now, now, job["id"]),
    )
    database.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=Path("data/translator.sqlite3"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    database = connect(args.db if args.db.is_absolute() else root / args.db)
    # PM2 can restart this single worker while an OCR child is using peak RAM.
    # A PROCESSING row then has no owner and would otherwise remain frozen
    # forever. On worker startup, safely return its sole interrupted job to the
    # queue; process_job is idempotent and rebuilds its image metadata/output.
    now = time.time()
    recovered = database.execute(
        "UPDATE jobs SET status='QUEUED',stage='RECOVERED_AFTER_RESTART',progress=0,"
        "updated_at=?,started_at=NULL,heartbeat_at=NULL,error=NULL "
        "WHERE status='PROCESSING'",
        (now,),
    ).rowcount
    database.commit()
    if recovered:
        print(f"Recovered {recovered} interrupted job(s)", flush=True)
    while True:
        minimum = int(setting(database, "minimum_free_ram_mb", os.getenv("TRANSLATOR_MIN_FREE_RAM_MB", "2500")))
        maximum_load = float(os.getenv("TRANSLATOR_PAUSE_LOAD_1M", "3.20"))
        job = claim_next(database, available_ram_mb(), minimum) if load_one_minute() <= maximum_load else None
        if job:
            try:
                # Resume is safe for both fresh and retried jobs. Completed
                # pages are verified and reused, so an automatic page retry
                # never restarts the chapter from page one.
                process_job(database, job, root, resume=True)
            except Exception as exc:
                # Branding failures must not crash/restart the worker and
                # silently requeue the same chapter in an endless loop.
                database.rollback()
                now=time.time()
                database.execute("UPDATE jobs SET status='FAILED',stage='FAILED',error=?,updated_at=?,completed_at=?,heartbeat_at=NULL WHERE id=? AND status='PROCESSING'",(str(exc)[:2000],now,now,job['id']))
                database.commit()
                print(f"Job {job['id']} failed: {exc}",flush=True)
            if args.once:
                return 0
            continue
        if args.once:
            return 0
        time.sleep(float(setting(database, "poll_interval_seconds", "5")))


if __name__ == "__main__":
    raise SystemExit(main())
