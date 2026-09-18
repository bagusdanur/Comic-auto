import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for page in sorted((root / "metadata").glob("page-*")):
    ocr = json.loads((page / "ocr-result.json").read_text())
    rows = [(round(r["confidence"], 3), r["text"], r["bbox"]) for r in ocr["regions"] if r["confidence"] < .75]
    if rows:
        print(page.name, json.dumps(rows, ensure_ascii=False))

