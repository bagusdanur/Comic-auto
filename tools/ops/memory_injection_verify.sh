#!/usr/bin/env bash
set -euo pipefail
cd /home/ryukomik/ryukomik-auto-comic-translator
comic_id="19b172ec-1dde-4548-a163-e3da8d19ba82"
job_id="093e5287-5c58-4772-a414-597bb3db73f2"
ocr_path="$(.venv/bin/python -c 'import sqlite3; c=sqlite3.connect("data/translator.sqlite3"); print(c.execute("select work_path from memory_jobs where id=?",("093e5287-5c58-4772-a414-597bb3db73f2",)).fetchone()[0]+"/pages/001/raw/ocr-result.json")')"
slug="$(.venv/bin/python -c 'import sqlite3; c=sqlite3.connect("data/translator.sqlite3"); print(c.execute("select slug from comics where id=?",("19b172ec-1dde-4548-a163-e3da8d19ba82",)).fetchone()[0])')"
.venv/bin/python translate_text_test.py "$ocr_path" --provider 9router --env-file .env.9router --memory "data/memory/$slug.json" --output /tmp/memory-smoke-check/translations.json >/dev/null
.venv/bin/python -c 'import json; d=json.load(open("/tmp/memory-smoke-check/translations.json")); print({"profile_used": d["memory_profile_used"], "terms_used": d["memory_terms_used"], "translation": d["groups"][0]["translation"]})'
