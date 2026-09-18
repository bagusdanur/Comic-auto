#!/usr/bin/env python3
"""Delete only explicitly supplied deployment smoke jobs and verify cleanup."""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("job_ids", nargs="+")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
token = (root / ".api-token").read_text(encoding="utf-8").strip()
headers = {"Authorization": f"Bearer {token}"}
for job_id in args.job_ids:
    url = f"http://127.0.0.1:3210/api/jobs/{job_id}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10) as response:
        job = json.load(response)
    job_root = Path(job["input_path"]).parent.parent
    request = urllib.request.Request(url, headers=headers, method="DELETE")
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.load(response)
    try:
        urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=10)
        status = 200
    except urllib.error.HTTPError as error:
        status = error.code
    print(json.dumps({"result": result, "get_after_delete": status, "storage_exists": job_root.exists()}))
    if status != 404 or job_root.exists():
        raise SystemExit("Delete verification failed")

