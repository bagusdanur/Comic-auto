#!/usr/bin/env python3
"""Re-clean and re-typeset pages previously hit by dense rectangle reconstruction."""
import json
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[2]
database = Path(sys.argv[1])
for job_root_arg in sys.argv[2:]:
    job_root = Path(job_root_arg)
    affected = []
    for page in sorted((job_root / "metadata").glob("page-*")):
        report = page / "removal-report.json"
        if not report.exists():
            continue
        results = json.loads(report.read_text(encoding="utf-8")).get("results", [])
        if not any(
            item.get("reason") == "DENSE_CAPTION_VERTICAL_RECONSTRUCTION"
            or (
                item.get("status") == "INPAINTED"
                and max((float(region.get("mask_ratio", 0)) for region in item.get("regions", [])), default=0) >= 0.38
            )
            for item in results
        ):
            continue
        pipeline = json.loads((page / "pipeline-status.json").read_text(encoding="utf-8"))
        surface = page / "surface-map.json"
        if surface.exists():
            subprocess.run([
                sys.executable, str(root / "quality_pipeline_v2.py"), "analyze",
                pipeline["input"], str(page / "ocr-result.json"),
                str(page / "translations.json"), "--output", str(surface),
            ], cwd=root, check=True, stdout=subprocess.DEVNULL)
        command = [
            sys.executable, str(root / "remove_text_test.py"), pipeline["input"],
            str(page / "ocr-result.json"), str(page / "translations.json"),
            "--output", str(page / "cleaned.png"),
            "--mask", str(page / "mask-debug.png"),
            "--report", str(page / "removal-report.json"),
        ]
        if surface.exists():
            command += ["--surface-map", str(surface)]
        subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
        affected.append(page.name)
    if not affected:
        continue
    job_id = job_root.name
    rerender = [
        sys.executable, str(root / "tools" / "repair" / "rerender_job_typeset.py"),
        str(database), str(job_root), job_id,
    ]
    for page_name in affected:
        rerender += ["--only-page", page_name]
    print(json.dumps({"job": job_id, "pages": affected}), flush=True)
    subprocess.run(rerender, cwd=root, check=True)

