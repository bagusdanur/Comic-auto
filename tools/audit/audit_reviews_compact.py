import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
for page in sorted((root / "metadata").glob("page-*")):
    translations_path = page / "translations.json"
    typeset_path = page / "typeset-report.json"
    quality_path = page / "quality-report.json"
    if not translations_path.exists() or not typeset_path.exists():
        continue
    groups = {item["id"]: item for item in json.loads(translations_path.read_text(encoding="utf-8"))["groups"]}
    typeset = json.loads(typeset_path.read_text(encoding="utf-8"))["results"]
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    review_ids = set(quality.get("review_ids", []))
    pipeline_path = page / "pipeline-status.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8")) if pipeline_path.exists() else {}
    pipeline_review = set(pipeline.get("review_ids", []))
    for item in typeset:
        if item.get("status") == "NEEDS_REVIEW":
            review_ids.add(item["id"])
    review_ids |= {item for item in pipeline_review if not str(item).startswith("quality:")}
    if not review_ids and not pipeline_review:
        continue
    print(page.name)
    for reason in sorted(item for item in pipeline_review if str(item).startswith("quality:")):
        print(json.dumps({"id": reason, "reason": reason}, ensure_ascii=False))
    by_type = {item["id"]: item for item in typeset}
    for group_id in sorted(review_ids):
        group = groups.get(group_id, {})
        item = by_type.get(group_id, {})
        source = " ".join(str(group.get("text", "")).split())
        target = " ".join(str(group.get("translation", "")).split())
        print(json.dumps({
            "id": group_id,
            "reason": item.get("reason", "QUALITY"),
            "bbox": group.get("bbox", item.get("bbox")),
            "font": item.get("font_size"),
            "preferred": item.get("preferred_font_size"),
            "source": source[:180],
            "translation": target[:240],
        }, ensure_ascii=False))

