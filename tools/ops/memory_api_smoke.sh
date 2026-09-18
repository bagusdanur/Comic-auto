#!/usr/bin/env bash
set -euo pipefail
cd /home/ryukomik/ryukomik-auto-comic-translator
.venv/bin/python make_memory_smoke.py
.venv/bin/python -m unittest -v test_queue.py
token="$(cat .api-token)"
title="Memory Smoke $(date +%s)"
comic="$(curl -fsS -H "Authorization: Bearer $token" -H 'Content-Type: application/json' -d "{\"title\":\"$title\"}" http://127.0.0.1:3210/api/comics)"
comic_id="$(printf '%s' "$comic" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
job="$(curl -fsS -H "Authorization: Bearer $token" -F chapterNumber=1 -F rawZip=@/tmp/memory-smoke/raw.zip -F translatedZip=@/tmp/memory-smoke/translated.zip "http://127.0.0.1:3210/api/comics/$comic_id/memory/analyze")"
job_id="$(printf '%s' "$job" | .venv/bin/python -c 'import json,sys; print(json.load(sys.stdin)["id"])')"
printf 'COMIC_ID=%s\nMEMORY_JOB_ID=%s\n' "$comic_id" "$job_id"
