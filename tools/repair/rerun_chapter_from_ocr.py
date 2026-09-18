import json
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
job = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)
originals = sorted((job / "original").glob("*"))
selected_pages = {int(value) for value in sys.argv[3:]} if len(sys.argv) > 3 else None
for index, source in enumerate(originals, 1):
    if selected_pages and index not in selected_pages:
        continue
    old = job / "metadata" / f"page-{index:03d}"
    page = destination / f"page-{index:03d}"
    page.mkdir(exist_ok=True)
    translations = page / "translations.json"
    cleaned = page / "cleaned.png"
    removal = page / "removal-report.json"
    final = destination / f"{index:03d}.png"
    commands = [
        [sys.executable, str(root / "translate_text_test.py"), str(old / "ocr-result.json"),
         "--provider", "9router", "--env-file", str(root / ".env.9router"),
         "--memory", str(destination / "memory.json"), "--output", str(translations)],
        [sys.executable, str(root / "remove_text_test.py"), str(source), str(old / "ocr-result.json"),
         str(translations), "--output", str(cleaned), "--mask", str(page / "mask.png"),
         "--report", str(removal)],
        [sys.executable, str(root / "typeset_test.py"), str(cleaned), str(translations),
         "--ocr-json", str(old / "ocr-result.json"), "--reference-image", str(source),
         "--removal-report", str(removal), "--output", str(final),
         "--report", str(page / "typeset-report.json")],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True, stdout=subprocess.DEVNULL)
    status = json.loads((page / "typeset-report.json").read_text())
    review = sum(row["status"] == "NEEDS_REVIEW" for row in status["results"])
    styles = {}
    for row in status["results"]:
        styles[row.get("style", "review")] = styles.get(row.get("style", "review"), 0) + 1
    print(json.dumps({"page": index, "review": review, "styles": styles}), flush=True)

