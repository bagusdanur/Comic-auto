import json
import sqlite3
import statistics
import sys
from collections import Counter
from pathlib import Path


database_path = Path(sys.argv[1])
title_query = sys.argv[2].casefold()
db = sqlite3.connect(database_path)
db.row_factory = sqlite3.Row
comic = db.execute("SELECT * FROM comics WHERE lower(title)=lower(?)", (sys.argv[2],)).fetchone()
if not comic:
    raise SystemExit(f"Comic not found: {sys.argv[2]}")
jobs = []
for chapter_dir in sorted((Path("data/jobs/library") / comic["slug"]).glob("chapter-*")):
    chapter_number = chapter_dir.name.removeprefix("chapter-")
    for metadata in chapter_dir.glob("*/metadata"):
        job = db.execute("SELECT * FROM jobs WHERE id=?", (metadata.parent.name,)).fetchone()
        if job:
            jobs.append((job, chapter_number, metadata))

for job, chapter_number, known_metadata in jobs:
    archive = Path(job["output_path"])
    chapter_root = archive.parent
    metadata_root = known_metadata
    # Production library layout has the job UUID one directory above output.
    if not metadata_root.exists():
        candidates = list(Path("data/jobs/library").glob(
            f"*/chapter-{chapter_number}/{job['id']}/metadata"
        ))
        metadata_root = candidates[0] if candidates else metadata_root
    statuses = Counter()
    reasons = Counter()
    fonts = []
    small = []
    translations = []
    page_count = 0
    for page in sorted(metadata_root.glob("page-*")):
        page_count += 1
        translation_path = page / "translations.json"
        report_path = page / "typeset-report.json"
        if not translation_path.exists() or not report_path.exists():
            continue
        groups = {g["id"]: g for g in json.loads(translation_path.read_text(encoding="utf-8"))["groups"]}
        for result in json.loads(report_path.read_text(encoding="utf-8"))["results"]:
            statuses[result.get("status", "UNKNOWN")] += 1
            if result.get("reason"):
                reasons[result["reason"]] += 1
            size = result.get("font_size")
            if size:
                fonts.append(size)
                group = groups.get(result["id"], {})
                translations.append({"source": group.get("text", ""), "target": group.get("translation", "")})
                if size < 20:
                    small.append({
                        "page": page.name, "id": result["id"], "font": size,
                        "preferred": result.get("preferred_font_size"),
                        "bbox": result.get("bbox"), "wrapped": result.get("wrapped"),
                        "source": group.get("text", ""), "target": group.get("translation", ""),
                    })
    print(json.dumps({
        "id": job["id"], "title": comic["title"], "chapter": chapter_number,
        "status": job["status"], "review_count": job["review_count"],
        "metadata": str(metadata_root), "pages": page_count,
        "statuses": statuses, "reasons": reasons,
        "font": ({"min": min(fonts), "median": statistics.median(fonts),
                  "under20": len(small), "count": len(fonts)} if fonts else {}),
        "small_samples": small[:30], "translations": translations,
    }, ensure_ascii=False))

