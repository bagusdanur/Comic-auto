import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = []
for page in sorted((root / "metadata").glob("page-*")):
    translations = json.loads((page / "translations.json").read_text())
    typeset = json.loads((page / "typeset-report.json").read_text())
    removal = json.loads((page / "removal-report.json").read_text())
    ocr = json.loads((page / "ocr-result.json").read_text())
    by_type = {item["id"]: item for item in typeset["results"]}
    by_remove = {item["id"]: item for item in removal["results"]}
    problems = []
    for group in translations["groups"]:
        rendered = by_type[group["id"]]
        cleaned = by_remove[group["id"]]
        source = group["text"].strip()
        target = group["translation"].strip()
        same = source.casefold() == target.casefold()
        if same or rendered["status"] != "TYPESET" or cleaned["status"] == "NEEDS_REVIEW":
            problems.append({
                "id": group["id"], "source": source, "translation": target,
                "same": same, "typeset": rendered["status"],
                "type_reason": rendered.get("reason"), "removal": cleaned["status"],
                "removal_details": cleaned.get("regions"),
                "bbox": group["bbox"], "font_size": rendered.get("font_size"),
                "preferred_font_size": rendered.get("preferred_font_size"),
            })
    summary.append({
        "page": page.name, "ocr_regions": ocr["region_count"],
        "low_confidence": ocr["low_confidence_count"],
        "groups": len(translations["groups"]), "problems": problems,
    })
print(json.dumps(summary, ensure_ascii=False, indent=2))

