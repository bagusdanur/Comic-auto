"""SQLite persistence and atomic claiming for the translator queue."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    requested_output_path TEXT,
    options_json TEXT NOT NULL DEFAULT '{}',
    stage TEXT NOT NULL DEFAULT 'QUEUED',
    progress REAL NOT NULL DEFAULT 0,
    review_count INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_processing_job
    ON jobs((1)) WHERE status = 'PROCESSING';
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    status TEXT NOT NULL,
    output_path TEXT,
    metadata_path TEXT,
    region_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS regions (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    source_text TEXT,
    translation TEXT,
    bbox_json TEXT NOT NULL,
    status TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS glossaries (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    entries_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS comics (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chapters (
    id TEXT PRIMARY KEY,
    comic_id TEXT NOT NULL REFERENCES comics(id) ON DELETE CASCADE,
    number TEXT NOT NULL,
    slug TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(comic_id, slug)
);
CREATE INDEX IF NOT EXISTS idx_chapters_comic ON chapters(comic_id, created_at);
CREATE TABLE IF NOT EXISTS memory_jobs (
    id TEXT PRIMARY KEY,
    comic_id TEXT NOT NULL REFERENCES comics(id) ON DELETE CASCADE,
    chapter_number TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT 'QUEUED',
    progress REAL NOT NULL DEFAULT 0,
    raw_zip_path TEXT NOT NULL,
    translated_zip_path TEXT NOT NULL,
    work_path TEXT NOT NULL,
    proposal_json TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL,
    heartbeat_at REAL
);
CREATE INDEX IF NOT EXISTS idx_memory_jobs_comic_created ON memory_jobs(comic_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_one_processing_memory_job
    ON memory_jobs((1)) WHERE status = 'PROCESSING';
CREATE TABLE IF NOT EXISTS memory_references (
    id TEXT PRIMARY KEY,
    comic_id TEXT NOT NULL REFERENCES comics(id) ON DELETE CASCADE,
    memory_job_id TEXT REFERENCES memory_jobs(id) ON DELETE SET NULL,
    chapter_number TEXT NOT NULL,
    summary_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_references_comic ON memory_references(comic_id, created_at DESC);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(path, timeout=30)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA busy_timeout=30000")
    database.executescript(SCHEMA)
    image_columns = {row[1] for row in database.execute("PRAGMA table_info(images)")}
    if "metadata_path" not in image_columns:
        database.execute("ALTER TABLE images ADD COLUMN metadata_path TEXT")
    now = time.time()
    defaults = {
        "concurrency": "1", "minimum_free_ram_mb": "2000",
        "poll_interval_seconds": "5",
    }
    for key, value in defaults.items():
        database.execute(
            "INSERT OR IGNORE INTO settings(key,value,updated_at) VALUES(?,?,?)",
            (key, value, now),
        )
    database.commit()
    return database


def row_dict(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    result = dict(row)
    if "options_json" in result:
        result["options"] = json.loads(result.pop("options_json"))
    return result


def claim_next(database: sqlite3.Connection, available_ram_mb: int, minimum_ram_mb: int) -> dict | None:
    now = time.time()
    database.execute("BEGIN IMMEDIATE")
    try:
        if database.execute("SELECT 1 FROM jobs WHERE status='PROCESSING' LIMIT 1").fetchone():
            database.commit()
            return None
        # Translation and memory training both load OCR models. Never let both
        # workers claim concurrently on the same VPS.
        memory_table = database.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memory_jobs'"
        ).fetchone()
        if memory_table and database.execute("SELECT 1 FROM memory_jobs WHERE status='PROCESSING' LIMIT 1").fetchone():
            database.commit()
            return None
        waiting = database.execute(
            "SELECT id FROM jobs WHERE status IN ('QUEUED','WAITING_RESOURCE') ORDER BY created_at LIMIT 1"
        ).fetchone()
        if not waiting:
            database.commit()
            return None
        if available_ram_mb < minimum_ram_mb:
            database.execute(
                "UPDATE jobs SET status='WAITING_RESOURCE',stage='WAITING_RESOURCE',updated_at=? WHERE id=?",
                (now, waiting["id"]),
            )
            database.commit()
            return None
        updated = database.execute(
            "UPDATE jobs SET status='PROCESSING',stage='STARTING',progress=0,started_at=?,updated_at=?,heartbeat_at=? "
            "WHERE id=? AND status IN ('QUEUED','WAITING_RESOURCE')",
            (now, now, now, waiting["id"]),
        )
        if updated.rowcount != 1:
            database.rollback()
            return None
        row = database.execute("SELECT * FROM jobs WHERE id=?", (waiting["id"],)).fetchone()
        database.commit()
        return row_dict(row)
    except Exception:
        database.rollback()
        raise


def new_id() -> str:
    return str(uuid.uuid4())
