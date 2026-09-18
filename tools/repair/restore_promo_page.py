import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from brand_postprocess import apply_branding
from webp_archive import archive_webp


parser = argparse.ArgumentParser()
parser.add_argument("database", type=Path)
parser.add_argument("job", type=Path)
parser.add_argument("job_id")
parser.add_argument("page", type=int)
args = parser.parse_args()
page_index = args.page - 1
images = {".png", ".jpg", ".jpeg", ".webp"}
originals = sorted(p for p in (args.job / "original").iterdir() if p.suffix.casefold() in images)
outputs = sorted(p for p in (args.job / "output").iterdir() if p.suffix.casefold() == ".png")
source, output = originals[page_index], outputs[page_index]
shutil.copyfile(source, output)
apply_branding(
    output, args.job / "metadata" / f"page-{args.page:03d}" / "ocr-result.json",
    Path(__file__).resolve().parents[2] / "assets/branding/watermark.png",
    Path(__file__).resolve().parents[2] / "assets/branding/watermark-text.png",
    first_page=page_index == 0,
    banner_path=(Path(__file__).resolve().parents[2] / "assets/branding/banner.png") if page_index == len(outputs) - 1 else None,
)
metadata = args.job / "metadata" / f"page-{args.page:03d}"
status_path = metadata / "pipeline-status.json"
status = json.loads(status_path.read_text(encoding="utf-8"))
status.update({"status": "COMPLETED", "review_ids": [], "passthrough": True,
               "passthrough_reason": "SOURCE_PROMO_ONLY"})
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
db = sqlite3.connect(args.database)
row = db.execute("SELECT id FROM images WHERE job_id=? AND metadata_path=?",
                 (args.job_id, str(metadata))).fetchone()
if row:
    db.execute("UPDATE images SET status='COMPLETED',output_path=? WHERE id=?", (str(output), row[0]))
    (args.job / "output" / "previews" / f"{row[0]}.jpg").unlink(missing_ok=True)
db.commit()
review_count = 0
for pipeline_path in sorted((args.job / "metadata").glob("page-*/pipeline-status.json")):
    review_count += len(json.loads(pipeline_path.read_text(encoding="utf-8")).get("review_ids", []))
archive_row = db.execute("SELECT output_path FROM jobs WHERE id=?", (args.job_id,)).fetchone()
if archive_row:
    archive_webp(outputs, Path(archive_row[0]))
db.execute("UPDATE jobs SET status=?,stage='DONE',progress=1,review_count=?,error=NULL WHERE id=?",
           ("REVIEW" if review_count else "COMPLETED", review_count, args.job_id))
db.commit()
print(f"page-{args.page:03d}: SOURCE_PROMO_ONLY")

