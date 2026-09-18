import json
import os
import subprocess
import sys
from pathlib import Path


root = Path(__file__).resolve().parents[2]
job = Path(sys.argv[1]).resolve()
for page_name in sys.argv[2:]:
    page = job / "metadata" / page_name
    pipeline = json.loads((page / "pipeline-status.json").read_text(encoding="utf-8"))
    command = [
        sys.executable, str(root / "remove_text_test.py"), pipeline["input"],
        str(page / "ocr-result.json"), str(page / "translations.json"),
        "--output", str(page / "cleaned.png"), "--mask", str(page / "mask-debug.png"),
        "--report", str(page / "removal-report.json"),
        "--hybrid-mode", os.getenv("HYBRID_PIPELINE", "xianscan"),
        "--xianscan-url", os.getenv("XIANSCAN_ML_URL", "http://127.0.0.1:8123"),
        "--hybrid-report", str(page / "hybrid-report.json"),
    ]
    surface = page / "surface-map.json"
    if surface.exists():
        command += ["--surface-map", str(surface)]
    subprocess.run(command, cwd=root, check=True)
    print(f"{page_name}: cleaned")

