#!/usr/bin/env bash
set -euo pipefail
root=/home/ryukomik/ryukomik-auto-comic-translator
cd "$root/api"
TRANSLATOR_DB="$root/data/phase7-smoke.sqlite3" \
TRANSLATOR_JOBS_DIR="$root/data/phase7-smoke-jobs" \
TRANSLATOR_API_PORT=3217 node dist/server.js >/tmp/phase7-api.log 2>&1 &
api_pid=$!
cleanup() {
  kill "$api_pid" 2>/dev/null || true
  wait "$api_pid" 2>/dev/null || true
  rm -f /tmp/phase7-status.json /tmp/phase7-upload.json
}
trap cleanup EXIT
for _ in 1 2 3 4 5; do
  if curl -fsS http://127.0.0.1:3217/api/system/status >/tmp/phase7-status.json; then break; fi
  sleep 1
done
curl -fsS -X POST \
  -F "file=@$root/samples/input.jpg" -F minY=1000 \
  http://127.0.0.1:3217/api/jobs >/tmp/phase7-upload.json
python3 - <<'PY'
import json
upload = json.load(open('/tmp/phase7-upload.json'))
status = json.load(open('/tmp/phase7-status.json'))
assert upload['status'] == 'QUEUED'
print('upload', upload['id'], upload['status'])
print('memory_available', status['memory']['available'], 'worker', status['worker']['status'])
open('/tmp/phase7-job-id', 'w').write(upload['id'])
PY
job_id=$(tr -d '\n' </tmp/phase7-job-id)
curl -fsS "http://127.0.0.1:3217/api/jobs/$job_id"
echo
curl -fsS -X POST "http://127.0.0.1:3217/api/jobs/$job_id/cancel"
echo
rm -f /tmp/phase7-job-id
