#!/usr/bin/env python3
"""Submit one known fixture to the production localhost API and verify output."""

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("source", nargs="?", type=Path)
parser.add_argument("--name")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
source = (args.source or root / "samples/input.jpg").resolve()
upload_name = args.name or source.name
token = (root / ".api-token").read_text(encoding="utf-8").strip()
boundary = f"----ryukomik-{uuid.uuid4().hex}"
payload = source.read_bytes()
body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{upload_name}\"\r\n"
    f"Content-Type: {'image/webp' if source.suffix.lower() == '.webp' else 'image/png' if source.suffix.lower() == '.png' else 'image/jpeg'}\r\n\r\n"
).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": f"multipart/form-data; boundary={boundary}",
}
request = urllib.request.Request("http://127.0.0.1:3210/api/jobs", data=body, headers=headers, method="POST")
with urllib.request.urlopen(request, timeout=30) as response:
    job = json.load(response)
job_id = job["id"]
print(json.dumps({"job_id": job_id, "submitted": True}), flush=True)
last = None
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    status_request = urllib.request.Request(
        f"http://127.0.0.1:3210/api/jobs/{job_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(status_request, timeout=15) as response:
        job = json.load(response)
    marker = (job["status"], job["stage"], job["progress"])
    if marker != last:
        print(json.dumps({"status": marker[0], "stage": marker[1], "progress": marker[2]}), flush=True)
        last = marker
    if job["status"] in {"COMPLETED", "REVIEW", "FAILED", "CANCELLED"}:
        break
    time.sleep(5)
else:
    raise SystemExit("Smoke job timed out")
if job["status"] not in {"COMPLETED", "REVIEW"}:
    raise SystemExit(f"Smoke job failed safely: {job.get('error')}")
download = urllib.request.Request(
    f"http://127.0.0.1:3210/api/jobs/{job_id}/download",
    headers={"Authorization": f"Bearer {token}"},
)
output = root / "data" / f"smoke-{job_id}{source.suffix.lower()}"
with urllib.request.urlopen(download, timeout=60) as response:
    output.write_bytes(response.read())
with Image.open(source) as original, Image.open(output) as translated:
    if original.size != translated.size:
        raise SystemExit(f"Dimension mismatch: {original.size} != {translated.size}")
    dimensions = original.size
print(json.dumps({
    "status": job["status"], "review_count": job["review_count"],
    "dimensions": dimensions, "output_bytes": output.stat().st_size,
}), flush=True)

