#!/usr/bin/env python3
"""Refresh V2 surface metadata and rerender only pages currently in review."""
import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
database = Path(sys.argv[1])
for job_arg in sys.argv[2:]:
    job = Path(job_arg)
    pages = []
    for page in sorted((job / "metadata").glob("page-*")):
        pipeline_path = page / "pipeline-status.json"
        if not pipeline_path.exists():
            continue
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if not pipeline.get("review_ids"):
            continue
        surface = page / "surface-map.json"
        subprocess.run([
            sys.executable, str(root / "quality_pipeline_v2.py"), "analyze",
            pipeline["input"], str(page / "ocr-result.json"),
            str(page / "translations.json"), "--output", str(surface),
        ], cwd=root, check=True, stdout=subprocess.DEVNULL)
        pages.append(page.name)
    if not pages:
        continue
    command = [
        sys.executable, str(root / "tools" / "repair" / "rerender_job_typeset.py"),
        str(database), str(job), job.name,
    ]
    for page_name in pages:
        command += ["--only-page", page_name]
    print(json.dumps({"job": job.name, "review_pages": pages}), flush=True)
    subprocess.run(command, cwd=root, check=True)

