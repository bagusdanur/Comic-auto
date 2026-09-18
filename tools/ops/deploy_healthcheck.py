#!/usr/bin/env python3
"""Authenticated localhost health check used during production deployment."""

import json
import urllib.error
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[2]
token = (root / ".api-token").read_text(encoding="utf-8").strip()
request = urllib.request.Request(
    "http://127.0.0.1:3210/api/system/status",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    status = json.load(response)
try:
    urllib.request.urlopen("http://127.0.0.1:3210/api/system/status", timeout=10)
    unauthorized = 200
except urllib.error.HTTPError as error:
    unauthorized = error.code
print(json.dumps({
    "authenticated": True,
    "unauthenticated_status": unauthorized,
    "worker": status["worker"],
    "memory_available": status["memory"]["available"],
    "queue": status["queue"],
}, indent=2))
if unauthorized != 401:
    raise SystemExit("Unauthenticated request was not rejected")

