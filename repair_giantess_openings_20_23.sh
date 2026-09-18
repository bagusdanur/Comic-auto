#!/usr/bin/env bash
set -euo pipefail
cd /home/ryukomik/ryukomik-auto-comic-translator
root=data/jobs/library/giantess-buddy-an-infinite-link-with-a-world-destroying-beauty
repair=data/repair/giantess-openings-20-23
backup=data/backups/giantess-opening-pages-before-repair-20260915
mkdir -p "$backup"

for chapter in $(seq 20 23); do
  job=$(find "$root/chapter-$chapter" -mindepth 1 -maxdepth 1 -type d | head -n 1)
  raw=$(find "$repair" -maxdepth 1 -type f -name "chapter-$chapter.*" | head -n 1)
  old=$(find "$job/original" -maxdepth 1 -type f -name '001_*' | head -n 1)
  out=$(find "$job/output" -maxdepth 1 -type f -name '001_*' | head -n 1)
  chapter_backup="$backup/chapter-$chapter"
  mkdir -p "$chapter_backup"
  cp -a "$old" "$out" "$job/metadata/page-001" "$chapter_backup/"
  cp "$raw" "$old"
  .venv/bin/python translate.py "$old" "$out" \
    --pipeline-engine v5 --provider 9router --env-file .env.9router --min-y 0 \
    --job-dir "$job/metadata/page-001" \
    --memory "data/memory/giantess-buddy-an-infinite-link-with-a-world-destroying-beauty.json" \
    --quality-v2 on --hybrid-mode xianscan-v3 --lama-model models/lama.onnx \
    --min-font-size 22 --min-scale-ratio 0.52 --layout-engine v5
  .venv/bin/python review_render.py archive "$job/output" "$job/output/chapter-translated.zip"
  echo "REPAIRED chapter-$chapter"
done
