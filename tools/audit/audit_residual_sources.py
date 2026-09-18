import json
import sys
from pathlib import Path

for job_arg in sys.argv[1:]:
    job = Path(job_arg)
    for quality_path in sorted((job / "metadata").glob("page-*/quality-report.json")):
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        candidates = quality.get("checks", {}).get("residual_candidates", [])
        if not candidates:
            continue
        translations = json.loads((quality_path.parent / "translations.json").read_text(encoding="utf-8"))
        groups = {item["id"]: item for item in translations.get("groups", [])}
        for candidate in candidates:
            group = groups.get(candidate["id"], {})
            print(json.dumps({
                "chapter": job.parent.name, "page": quality_path.parent.name,
                "id": candidate["id"], "ratio": candidate.get("unchanged_foreground_ratio"),
                "source": group.get("text"), "target": group.get("translation"),
            }, ensure_ascii=False))

