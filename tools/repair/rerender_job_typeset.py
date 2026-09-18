#!/usr/bin/env python3
"""Re-typeset a completed batch from cached OCR/translation/cleanup artifacts."""
import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

from brand_postprocess import apply_branding
from quality_pipeline_v2 import validate
from translate import is_promo_only_page
from webp_archive import archive_webp

parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("job_root", type=Path)
parser.add_argument("job_id")
parser.add_argument("--min-font-size", type=int, default=22)
parser.add_argument("--min-scale-ratio", type=float, default=.52)
parser.add_argument("--only-page", action="append", default=[])
parser.add_argument("--branding-enabled", action="store_true")
parser.add_argument("--layout-engine", choices=("v3", "v4", "v5"), default="v3")
args = parser.parse_args()
args.job_root = args.job_root.resolve()
args.database = args.database.resolve()

root = Path(__file__).resolve().parents[2]
pages = sorted((args.job_root / "metadata").glob("page-*"))
outputs = sorted((args.job_root / "output").glob("*.png"))
if len(outputs) != len(pages):
    raise SystemExit(f"Expected one PNG master per page: pages={len(pages)} outputs={len(outputs)}")
database = sqlite3.connect(args.database, timeout=30)
database.row_factory = sqlite3.Row
review_count = 0

for index, (page, output) in enumerate(zip(pages, outputs)):
    pipeline_path = page / "pipeline-status.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("passthrough"):
        ocr_path = page / "ocr-result.json"
        translations_path = page / "translations.json"
        no_safe_groups = bool(
            translations_path.exists()
            and not json.loads(translations_path.read_text(encoding="utf-8")).get("groups")
        )
        if ocr_path.exists() and no_safe_groups:
            pipeline.update({"status": "COMPLETED", "review_ids": [],
                             "passthrough_reason": (
                                 "SOURCE_PROMO_ONLY" if is_promo_only_page(
                                     json.loads(ocr_path.read_text(encoding="utf-8"))
                                 ) else "NO_SAFE_DIALOGUE_PASSTHROUGH"
                             )})
            pipeline_path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
            image = database.execute("SELECT id FROM images WHERE job_id=? AND metadata_path=?",
                                     (args.job_id, str(page))).fetchone()
            if image:
                database.execute("UPDATE images SET status=? WHERE id=?", ("COMPLETED", image["id"]))
                database.commit()
            continue
        # Old promo-only classification can be stale when the same page also
        # contains valid translated dialogue. Clear it and render those groups.
        pipeline.pop("passthrough", None)
        pipeline.pop("passthrough_reason", None)
    if args.only_page and page.name not in args.only_page:
        continue
    command = [
        sys.executable, str(root / "typeset_test.py"), str(page / "cleaned.png"),
        str(page / "translations.json"), "--ocr-json", str(page / "ocr-result.json"),
        "--reference-image", pipeline["input"], "--removal-report", str(page / "removal-report.json"),
        "--output", str(page / "typeset.png"), "--report", str(page / "typeset-report.json"),
        "--min-font-size", str(args.min_font_size), "--min-scale-ratio", str(args.min_scale_ratio),
        "--layout-engine", args.layout_engine,
    ]
    surface = page / "surface-map.json"
    if surface.exists():
        command += ["--surface-map", str(surface), "--hard-clip"]
    subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
    quality = None
    if surface.exists():
        quality = validate(Path(pipeline["input"]), page / "cleaned.png", page / "typeset.png",
                           page / "mask-debug.png", surface, page / "quality-report.json")
    shutil.copyfile(page / "typeset.png", output)
    branding = (
        apply_branding(
            output, page / "ocr-result.json", root / "assets/branding/watermark.png",
            root / "assets/branding/watermark-text.png", first_page=index == 0,
            banner_path=(root / "assets/branding/banner.png") if index == len(pages) - 1 else None,
        )
        if args.branding_enabled
        else {"enabled": False, "reason": "BRANDING_DISABLED"}
    )
    (page / "branding-report.json").write_text(json.dumps(branding, indent=2), encoding="utf-8")
    typeset = json.loads((page / "typeset-report.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in typeset.get("results", [])}
    review_ids = [item["id"] for item in by_id.values() if item.get("status") == "NEEDS_REVIEW"]
    if quality and quality.get("status") != "PASS":
        review_ids += [f"quality:{reason}" for reason in quality.get("review_reasons", [])]
    review_ids = sorted(set(review_ids))
    pipeline.update({"status": "REVIEW" if review_ids else "COMPLETED", "review_ids": review_ids,
                     "pipeline_engine": args.layout_engine, "layout_engine": args.layout_engine,
                     "quality_v2_mode": "on", "quality_v2": quality})
    pipeline_path.write_text(json.dumps(pipeline, indent=2), encoding="utf-8")
    image = database.execute("SELECT id FROM images WHERE job_id=? AND metadata_path=?",
                             (args.job_id, str(page))).fetchone()
    if image:
        (output.parent / "previews" / f"{image['id']}.jpg").unlink(missing_ok=True)
        database.execute("UPDATE images SET status=?,output_path=? WHERE id=?",
                         (pipeline["status"], str(output), image["id"]))
        for region in database.execute("SELECT id,metadata_json FROM regions WHERE image_id=?", (image["id"],)):
            old = json.loads(region["metadata_json"] or "{}")
            replacement = by_id.get(old.get("id"))
            if replacement:
                database.execute("UPDATE regions SET status=?,metadata_json=? WHERE id=?",
                                 (replacement["status"], json.dumps(replacement), region["id"]))
    database.commit()
    print(f"{page.name}: {pipeline['status']} ({len(review_ids)} review)", flush=True)

review_count = sum(len(json.loads((page / "pipeline-status.json").read_text(encoding="utf-8")).get("review_ids", []))
                   for page in pages)
job_row = database.execute(
    "SELECT output_path,options_json FROM jobs WHERE id=?", (args.job_id,)
).fetchone()
archive_path = Path(job_row["output_path"])
members = archive_webp(outputs, archive_path)
final_status = "REVIEW" if review_count else "COMPLETED"
options = json.loads(job_row["options_json"] or "{}")
options.update({"pipeline_engine": args.layout_engine, "layout_engine": args.layout_engine})
database.execute("UPDATE jobs SET status=?,stage='DONE',progress=1,review_count=?,error=NULL,options_json=? WHERE id=?",
                 (final_status, review_count, json.dumps(options), args.job_id))
database.commit()
print(json.dumps({"status": final_status, "review_count": review_count, "webp_members": members}))

