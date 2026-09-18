from pathlib import Path
from PIL import Image
from zipfile import ZipFile
import json

root = Path("data/jobs/library/giantess-buddy-an-infinite-link-with-a-world-destroying-beauty")
repair = Path("data/repair/giantess-openings-20-23")
for chapter in range(20, 24):
    job = next((root / f"chapter-{chapter}").iterdir())
    raw = next(repair.glob(f"chapter-{chapter}.*"))
    original = next((job / "original").glob("001_*"))
    output = next((job / "output").glob("001_*"))
    with Image.open(raw) as a, Image.open(original) as b, Image.open(output) as c:
        dimensions = (a.size, b.size, c.size)
    quality_path = job / "metadata/page-001/quality-report.json"
    quality = json.loads(quality_path.read_text()).get("status") if quality_path.exists() else None
    zip_path = job / "output/chapter-translated.zip"
    with ZipFile(zip_path) as archive:
        bad = archive.testzip()
        count = len(archive.namelist())
    print(chapter, "dims", dimensions, "quality", quality, "zip", count, "bad", bad)
