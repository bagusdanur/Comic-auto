# HANDOFF — ryukomik-auto-comic-translator

Ditulis 2026-10-21 dari kode repo + git + riwayat session, lalu **disinkronkan ulang 2026-10-21 09:54**
terhadap `git status`, isi `data/translator.sqlite3`, dan hasil `pytest` yang benar-benar dijalankan. Bagian yang tidak bisa
dibuktikan ada di §10 — jangan diisi tebakan.

Catatan istilah: **semua perubahan belum di-commit di repo ini adalah WIP session ini** (fitur yang
sedang dikerjakan), bukan pekerjaan orang lain. Jangan dianggap sampah, jangan di-revert.

## 1. Ringkasan

- Proyek: pipeline OCR + AI translator komik/webtoon Inggris → Indonesia untuk Ryukomik, jalan di VPS ini.
- Input: file raw per halaman (`data/jobs/.../original/*.avif|webp|png`) atau satu gambar panjang.
- Alur: OCR (deteksi teks + bubble) → translasi → analisis surface → hapus teks lama (inpaint hybrid)
  → typeset teks Indonesia → QC → export PNG/WebP + ZIP chapter.
- Bagian OCR/AI dipakai dua tempat: (a) OCR lokal PaddleOCR membaca teks Inggris dan bbox-nya,
  (b) LLM (9router) menerjemahkan per grup dialog jadi Indonesia.
- Semua tahap nulis artefak per halaman ke `metadata/page-NNN/`; `pipeline-status.json` adalah
  sumber status/`stage` yang dibaca dashboard.

## 2. Arsitektur runtime

PM2 (definisi: `ecosystem.config.cjs`, dijalankan dari root repo) — keempatnya `online` saat handoff:

| Proses | Port | Entry point | Catatan |
|---|---|---|---|
| `ryukomik-xianscan-ml` | 8123 (127.0.0.1) | `vendor/xianscan/ml/.venv/bin/uvicorn --app-dir <root> xianscan_sidecar:app` | sidecar inpaint; model di `vendor/xianscan/ml/models/`: `lama.onnx` **197 MB**, `comictextdetector.pt.onnx` **91 MB** (total dir 288 MB). `/health` → `inpainter: lama-onnx`, `ocr: rapidocr` |
| `ryukomik-translator-api` | 3210 (127.0.0.1) | `api/dist/start.js` (node, Fastify; source `api/src/server.ts`) | REST untuk dashboard, `max_memory_restart: 180M` |
| `ryukomik-translator-worker` | — | `queue_worker.py` (`.venv/bin/python`) | pemroses antrean, `max_memory_restart: 2400M`, `concurrency=1` (restart terakhir: 2x) |
| `ryukomik-memory-worker` | — | `memory_trainer.py` | belajar istilah/register Indonesia ke tabel memory, `max_memory_restart: 1500M` |

- DB: **`data/translator.sqlite3`** (env `TRANSLATOR_DB`). Tabel: `comics`, `chapters`, `jobs`,
  `images`, `regions`, `glossaries`, `settings`, `memory_jobs`, `memory_references`.
- `jobs`: `id,status,stage,progress,input_path,output_path,options_json,review_count,error,...`;
  `images`: `id,job_id,filename,status,output_path,metadata_path,region_count,crop_top,crop_bottom`.
- Job dir: `data/jobs/<library>/<judul>/chapter-N/<job_id>/` → `original/`, `prepared-joined-v1/`,
  `metadata/page-NNN/`, `output/` (`<stem>_<stem>.png`, `chapter-translated.zip`).
- Auth API: header `Authorization: Bearer $(cat .api-token)` (token 65 byte di root repo, gitignored,
  dibaca `api/src/server.ts`, juga dipakai `ecosystem.config.cjs` → `TRANSLATOR_API_TOKEN`).
  **Jangan tulis token ke dokumen/log.**
- Endpoint utama: `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/:id`,
  `POST /api/jobs/:id/cancel`, `GET /api/queue`, `POST /api/images/:id/render` (edit bubble),
  `GET /api/jobs/:id/download`, `GET /api/settings`, `POST /api/settings/test-translator`,
  `GET /api/system/status`.
- Provider model per tahap:
  - OCR: **PaddleOCR lokal CPU** (`translate_test.py:23`), atau `resident_ocr.py` kalau job
    multi-halaman (`queue_worker.py:82-86`; `TRANSLATOR_OCR_RESIDENT` default `'1'`, aktif bila input >1 file).
  - Translasi: `--provider 9router` default (`queue_worker.py:136`), env `.env.9router`, base URL
    `http://127.0.0.1:20128/v1` (9Router lokal). **Model aktif diambil dari tabel `settings`**
    (`ninerouter_model`, `ninerouter_fallback_models`) dan menimpa env (`queue_worker.py:163-167`).
  - Inpaint: sidecar XianScan (LaMa ONNX) + fallback deterministik OpenCV
    (`remove_text_test.py:526,545,1118`).
- Admission resource: `queue_db.claim_next()` menolak klaim job bila disk kurang
  (`disk_budget.disk_check`) → status `WAITING_RESOURCE` / `stage: WAITING_DISK`; RAM gate lewat
  `TRANSLATOR_MIN_FREE_RAM_MB=2200`.

## 3. Alur OCR/AI end-to-end

Dijalankan `queue_worker.py` → `process_job()` membuka konteks `resident_ocr.chapter_ocr()`
→ subprocess `translate.py` per halaman (`queue_worker.py:128-150`):

1. **JOIN** — `priority_seams.py` → `prepare_inputs`, hasil `prepared-joined-v1/manifest.json`.
   Mode join bisa dipaksa lewat `job_queue.py enqueue --raw-mode auto|original|joined`
   (disimpan di `options.raw_input_mode`); `auto` = hanya judul prioritas yang di-join.
2. **OCR** (`stage: OCR`) — `translate_test.py` → `metadata/page-NNN/ocr-result.json`
   (regions + bbox + confidence). Job multi-halaman memakai satu proses OCR resident per chapter.
3. **TRANSLATION** (`stage: TRANSLATION`) — `translate_text_test.py` → `translations.json`
   (`groups[]` = hasil merge fragmen + `text`/`translation`). Provider: `translation_providers.py`
   (`NineRouterProvider` :311, `DeepSeek` :220, `Gemini` :254, `OpenAICompatible` :278; factory :465).
4. **SURFACE_ANALYSIS_V2** — `quality_pipeline_v2.py analyze` → `surface-map.json`
   (jenis surface: bubble/caption/artwork, `source_credit`).
5. **REMOVAL** (`stage: REMOVAL`) — `remove_text_test.py` → `cleaned.png`, `removal-report.json`,
   `mask-debug.png`, `hybrid-report.json`. Mask via `footprint_masks.py`, `flat_glyph_fill_mask`,
   `enclosed_bubble.py`, `verified_flat_cleanup.py`; hybrid via `xianscan_client.py` + `hybrid_inpaint.py`.
6. **TYPESETTING** — `typeset_test.py` (engine `typeset_engine_v5.py`) → `typeset.png`,
   `typeset-report.json` (status per grup), plus `source_typography.py` / `source_text_color.py`
   untuk mengikuti gaya huruf & warna sumber.
7. **QUALITY_V2 validate** — `quality-report.json` (gagal → `REVIEW`, id `quality:<reason>`).
8. **EXPORT** — PNG ke `output/`, lalu `webp_archive.py` bikin `chapter-translated.zip`;
   status akhir `COMPLETED` atau `REVIEW` (`translate.py:304`).
9. Editor bubble live: `POST /api/images/:id/render` → `review_render.py`.

Status sah per grup di `typeset-report.json`: `TYPESET`, `PRESERVED`, `NEEDS_REVIEW`
(`typeset_engine_v5.py:619`); `PRESERVED` dengan `reason: UNFIT_SOURCE_RESTORED` artinya teks Inggris
dikembalikan karena tidak muat.

## 4. Keputusan FINAL (jangan diubah tanpa bukti baru)

- **Hybrid = `xianscan-v3` lewat sidecar, bukan LaMa lokal.** `HYBRID_PIPELINE=xianscan-v3` di
  `ecosystem.config.cjs`. `models/lama.onnx` di root repo **tidak ada** (dir `models/` kosong),
  jadi mode `on` lokal tidak jalan; sidecar yang punya `lama.onnx` (197 MB).
  Bukti produksi: `metadata/page-005/removal-report.json` → `mode: xianscan-v3`,
  `sidecar_inpaint_batches: 4`, backend `v5-source-footprint-lama-batched`.
- **Engine v5 adalah produksi.** `PIPELINE_ENGINE=v5`, `TYPESET_LAYOUT_ENGINE=v5`.
  `typeset_engine_v4.py`/`retypeset_job_v4.py` hanya historis.
- **Mask tidak boleh melewati garis bubble.** `enclosed_bubble.py:36-39`: pertumbuhan mask dibatalkan
  kalau menyentuh tepi crop atau >1.8× area komponen ketat. Ambang longgar `188/62` (min channel /
  peak-to-peak) dipilih dari uji: `180/50` bocor (bounds 1558 vs bubble 1490).
- **White-fill hanya untuk komponen huruf.** `flat_glyph_fill_mask` di `remove_text_test.py`;
  `verified_white_fill` (kotak persegi) menghasilkan kotak putih yang memotong busur bubble.
  Varian "hanya glyph putih polos" yang aman ada di `verified_flat_cleanup.py`.
- **Komponen memanjang bukan glyph.** `elongated_line` (`w > max(14,h*9)` / `h > max(18,w*12)`)
  di `remove_text_test.py`; busur outline lebih tipis-panjang dari satu huruf.
- **Anchor teks: coba anchor sumber dulu, kalau mentok pakai tengah pita mask** (`_vertical_band_center`,
  pita ≥55% lebar maks). Alasan: anchor dari glyph asli sering duduk di dasar bubble → teks nempel atap.
- **Line spacing = 5% font** (`max(1, round(size*.05))`), bukan 12%. 12% terbaca renggang di bubble besar.
- **Kalau tidak muat, teks Inggris dikembalikan** (`PRESERVED / UNFIT_SOURCE_RESTORED`), bukan font
  dikecilkan sampai tak terbaca dan bukan ditinggal kosong.
- **Urutan baca fragmen** pakai `ROW_BUCKET=12` + `_reading_order()` di `translate_text_test.py`;
  sort `(y,x)` mentah salah untuk kata sebaris. Fragmen vertikal digabung lewat
  `merge_vertical_fragments()`, dan glyph yang sangat miring dipisah sebelum grouping bbox
  (komentar: "Separate strongly rotated glyphs from horizontal labels").
- **KEN COMICS/SCANS hanya kredit kalau geometri terverifikasi.** `branding_v51.confirmed_credit()`
  menolak teks yang match `KEN COMICS|SCANS` bila `kencomics_mark.is_kencomics()` tidak fullmatch;
  mask capsule/avatar/fox-nya dari `kencomics_mark.kencomics_mask()` (butuh caption putih-di-hitam
  terukur, aspect 2–9, atau raise `RuntimeError`). V5.2 menambahkan `dusk_pill_mask()` untuk
  capsule Luster — pakai emblem merah + caption OCR, bukan ambang gelap global.
- **OCR resident, bukan proses per halaman.** `resident_ocr.py`: satu proses OCR job-scoped dengan
  model PaddleOCR yang dipakai ulang; dipakai kalau input >1 file. Memangkas biaya load model per halaman.
- **Model referensi/memory memakai geometri artwork, bukan nomor halaman ZIP.**
  `memory_alignment.py` menyelaraskan halaman referensi lewat artwork; `reference_memory.py`
  memilih konteks chapter yang bounded; `pronoun_policy.py` hanya menerapkan pilihan pronominal
  yang dikunci per judul. Jangan menyimpulkan alignment dari urutan ZIP.
- **Worker RAM 2400M** karena 1500M mematikan worker di tengah halaman 60k px (222 restart beruntun).
  `TRANSLATOR_MIN_FREE_RAM_MB=2200`, `concurrency=1`. Memory-worker cukup 1500M.
- **Jangan `git add -A`.** Working tree berisi 46 entri (19 modified + 27 untracked) milik session ini;
  commit harus per path eksplisit dan per topik.

## 5. Jebakan / known issues (semua pernah kejadian)

1. **Gejala: setelah ubah kode, output halaman tidak berubah.**
   Penyebab: resume memakai `pipeline-status.json` + baris `images` berstatus COMPLETED
   (`queue_worker.py:117-131`, cetak `REUSED COMPLETED`).
   Aman: render satu halaman → `tools/repair/rerender_job_typeset.py <db> <job_dir> <job_id> --only-page page-00X`;
   job penuh → **backup dulu** (`cp data/translator.sqlite3 data/backup/translator-$(date +%s).sqlite3`
   dan `cp -a metadata metadata.bak`), baru hapus `pipeline-status.json`,
   `DELETE FROM images WHERE job_id=...`, `UPDATE jobs SET status='QUEUED'`, lalu
   `tools/repair/requeue_job.py`.
2. **Gejala: teks nempel di atap bubble walau ada ruang bawah.**
   Penyebab: anchor tengah glyph sumber ada di tepi mask (contoh page-005 `bubble-002`: anchor y=258.5
   di mask 268 px). Aman: `_vertical_band_center` di `typeset_engine_v5.py`. Jangan menaikkan
   `spacing` untuk "mengisi".
3. **Gejala: mask bubble berhenti di tengah, dasar bubble tak terpakai.**
   Penyebab: interior bubble bergradasi (dasar ~(194,186,207)) gagal ambang ketat `min>215`.
   Aman: growth di `enclosed_bubble.py:30-39` dengan dua guard di atas. Jangan melonggarkan tanpa
   cek `bounds` vs tepi bubble.
4. **Gejala: kotak putih persegi menembus busur bubble.**
   Penyebab: white-fill memakai footprint bbox. Aman: `flat_glyph_fill_mask` + `elongated_line`
   (test: `tests/test_flat_glyph_fill.py`, `tests/test_region_mask_outline.py`,
   `tests/test_verified_flat_cleanup.py`).
5. **Gejala: satu kalimat pecah jadi beberapa bubble.**
   Penyebab: fragmen baris tidak digabung / urutan salah. Aman: merge + `_parts` + `_reading_order`
   (`tests/test_translation_memory.py::SentenceContinuationTests`).
6. **Gejala: satu halaman tidak diterjemahkan sama sekali.**
   Penyebab: semua kandidat `UNFIT` → `PRESERVED / UNFIT_SOURCE_RESTORED`. Aman: cek
   `typeset-report.json`, jangan diterjemahkan manual lewat edit DB.
7. **Gejala: `pytest tests` gagal di `test_branding_v51::test_evascans_source_variants_are_confirmed`.**
   **Ini bukan regresi WIP** — sudah merah juga di HEAD `5373dee` (dibuktikan dengan menjalankan
   `branding_v51.py` versi HEAD). Penyebab asli: `ANCHOR` di `branding_v51.py:21` memuat
   `eva\s*scans?`, sehingga kalimat "Eva scans the room carefully" (5 kata ≤ 14) dianggap kredit.
   Perbaikan yang benar ada di deteksi `ANCHOR`/fallback, **bukan** dengan mengubah asersi.
8. **Gejala: sidecar XianScan mati → removal diam-diam jatuh ke OpenCV.**
   Cek `curl 127.0.0.1:8123/health`; `pm2 restart ryukomik-xianscan-ml`.
   Laporan `hybrid-report.json` menunjukkan `sidecar_inpaint_batches: 0` (bukan error).
9. **Gejala: teks Indonesia tetap ada teks Inggris sisa.**
   Penyebab historis: OCR region dobel → dedup memilih versi per-kata → orphan. Fix containment
   sudah ada (commit `adc42d9`); kalau muncul lagi cek `removal-report.json` `results[].status`.
10. **Gejala: diff besar sekali padahal logika sedikit berubah.**
    Penyebab: sebagian file tertulis ulang dengan **CRLF**. Bukti: `translate_text_test.py` tampak
    949 baris berubah, tapi 93 saja setelah `git diff --ignore-cr-at-eol`; total repo 1904/1186
    turun jadi **825 insertions / 107 deletions**. Normalisasi line ending SEBELUM commit, kalau tidak
    review jadi tidak mungkin dan `git blame` rusak.
11. **Gejala: job tidak pernah diambil worker.**
    Cek `jobs.status`/`jobs.stage`: `WAITING_RESOURCE` + `WAITING_DISK` artinya `disk_budget.disk_check`
    menolak (bukan bug worker). Bebaskan disk atau naikkan batas; jangan paksa status ke QUEUED
    tanpa cek `disk_health` (`tools/ops/storage_health.py`).
12. **DITERIMA apa adanya: sisa ~21 px teks agak ke atas di page-005** (mask growth berhenti ~42 px di
    atas garis bubble; perbaikannya butuh deteksi kontur outline, bukan pelonggaran ambang).
    Job `7cfe2ac0` sudah dinyatakan **SELESAI** oleh user (§6) — **dibiarkan oleh user, jangan diubah
    tanpa perintah baru**.

## 6. Status terakhir (2026-10-21 09:54 WIB)

Latar: handoff awal ditulis 09:30. Bagian di bawah ini **diverifikasi ulang 2026-10-21 09:54 WIB**
dengan menjalankan perintah, bukan dari catatan session:

- **Job `7cfe2ac0` SELESAI** — dinyatakan selesai oleh **user**, bukan auto-QC. Diverifikasi 09:54 WIB
  langsung dari `data/translator.sqlite3` (`.venv/bin/python`, bukan sqlite3 CLI):
  `status=COMPLETED`, `stage=DONE`, `progress=1.0`, `error=NULL`,
  `completed_at=2026-10-21 08:24:47 WIB`; tabel `images` → 15 baris `COMPLETED`.
  Judul/path: `data/jobs/library/pyscho-apartment/chapter-1/7cfe2ac0-.../` (Pyscho Apartment ch1, 15 halaman).
  Tally hasil tetap **155 `TYPESET` / 29 `PRESERVED` / 0 `UNFIT` / 0 review**.
- **Test suite**: `.venv/bin/python -m pytest tests -q` → **347 passed, 1 failed, 59 subtests, 23.65 s**
  (dijalankan saat handoff 09:30; tidak dijalankan ulang pada sinkronisasi ini).
  Yang gagal hanya `test_branding_v51` (lihat §5.7 — sudah merah di HEAD, bukan akibat WIP).
- **Service**: 4 proses PM2 `online`; API `/api/system/status` → HTTP 200; sidecar `/health` → `ok`,
  `inpainter: lama-onnx`, `accelerator: CPU Multi-threaded`.
- **Model translasi aktif** (dari DB, bukan kode): `settings.ninerouter_model = ag/gemini-3.8-flash-low`,
  fallback `ag/claude-sonnet-4-6,cbai/deepseek-v4.1-flash`.
- **Selesai & ter-deploy**: perbaikan posisi teks + spacing + mask bubble page-005
  (font 27/5 baris → 32/6 baris); `output/005_005.png` dan ZIP sudah diperbarui; worker sudah di-restart.
- **DIBATALKAN atas keputusan user**: render ulang 15 halaman penuh / page-006..015 untuk mengukur
  efek mask growth. Job sudah SELESAI, jadi **jangan dijalankan** (lihat §8.2).
- **Sengaja dimatikan, bukan bug**: logo/emblem branding yang sudah ada di halaman sumber
  dipertahankan (`PRESERVED`, mis. `DEFER_SOURCE_BRANDING`); promo/credit page di-passthrough
  (`SOURCE_PROMO_ONLY`).

### 6.1 Semua perubahan belum di-commit (WIP session ini, 46 entri)

HEAD = commit docs-only sinkronisasi ini (`git log -1`; isinya **hanya** `docs/HANDOFF.md`). Sebelumnya
`ab51e4d` ("docs: sinkronkan handoff dengan seluruh WIP (47 entri)"); basis **kode** masih `5373dee`
("Bersihkan hanya huruf: mask tidak boleh memotong garis bubble") — belum ada commit kode sejak itu.
`git status --porcelain` = **46 entri: 19 `M` + 27 `??`** (diukur 2026-10-21 09:54 WIB, sesudah commit
docs ini; `docs/HANDOFF.md` sudah bersih, bukan bagian dari 46).

Ukuran nyata per area (angka = baris berubah setelah CR/whitespace diabaikan):

| Area | File | Isi |
|---|---|---|
| Mask & typeset | `enclosed_bubble.py` (+16), `typeset_engine_v5.py` (+134), `footprint_masks.py` (baru, 30), `verified_flat_cleanup.py` (baru, 37) | growth guard bubble, anchor pita, mask footprint hemat memori, cleanup glyph putih polos |
| Branding | `branding_v51.py` (+5), `branding_v52.py` (+88), `kencomics_mark.py` (baru, 30) | verifikasi KEN, capsule Luster, mask signature |
| OCR | `resident_ocr.py` (baru, 179) | proses OCR job-scoped, model dipakai ulang |
| Translasi teks | `translate_text_test.py` (+93), `memory_trainer.py` (+131) | grouping/rotated-glyph/reading-order, chunking chapter penuh |
| Sumber teks | `source_typography.py` (+25), `source_text_color.py` (+114) | italic reconciliation, harmonisasi palet, ukur efek glyph |
| Referensi & memory | `reference_memory.py` (baru, 113), `memory_alignment.py` (baru, 64), `pronoun_policy.py` (baru, 23), `api/src/memory-reference.ts` (baru, 38) + `api/src/memory-reference.test.ts`, `api/src/memory-upload.test.ts` (baru) | grounding referensi, alignment via artwork, pronominal terkunci |
| Input & antrean | `raw_join.py` (baru, 73), `queue_worker.py` (+24), `job_queue.py` (+3), `queue_db.py` (+11), `translate.py` (+8) | join tanpa resize, `--raw-mode`, admission disk |
| Storage | `disk_budget.py` (baru, 50), `tools/ops/{maintain_storage,retain_deploy_backups,storage_health}.py` (baru) | cek disk, retensi backup, health |
| API | `api/src/server.ts` (+63) | endpoint memory reference/upload |
| Test baru (untracked) | `tests/test_{backup_retention,disk_budget,enclosed_bubble_gradient,group_geometry,indonesian_memory_worker,luster_capsule,memory_alignment,pronoun_policy,reference_memory,resident_ocr,storage_maintenance,verified_flat_cleanup}.py` | 12 file |
| Test diubah | `test_mask_center_anchor.py`, `test_source_text_color.py`, `test_source_typography.py`, `test_translation_memory.py`, `test_typeset_engine_v5.py` | 5 file |
| Vendor | `vendor/xianscan` (submodule/isi berubah) | model & kode sidecar |

`docs/HANDOFF.md` sudah ter-commit di `ab51e4d` (docs-only) — **tidak** lagi bagian dari 46 entri WIP.
Commit sinkronisasi berikutnya juga docs-only, path eksplisit (`git commit -- docs/HANDOFF.md`).

## 7. Verifikasi cepat

```bash
cd /home/ryukomik/ryukomik-auto-comic-translator
.venv/bin/python -m pytest tests -q                 # ~21 s; saat handoff: 347 passed, 1 failed (§5.7)
.venv/bin/python -m pytest tests/test_mask_center_anchor.py -q
TOKEN=$(cat .api-token)
curl -s -H "Authorization: Bearer $TOKEN" http://127.0.0.1:3210/api/system/status
curl -s http://127.0.0.1:8123/health | head -c 300   # sidecar inpaint
pm2 list | grep -E "translator|xianscan|memory"
pm2 logs ryukomik-translator-worker --lines 80 --nostream
pm2 logs ryukomik-translator-api --lines 40 --nostream
.venv/bin/python -c "import sqlite3;c=sqlite3.connect('data/translator.sqlite3');print(*c.execute('select id,status,stage,progress from jobs order by created_at desc limit 3'),sep=chr(10))"
.venv/bin/python -c "import sqlite3;c=sqlite3.connect('data/translator.sqlite3');print(*c.execute(\"select key,value from settings where key like 'ninerouter%'\"),sep=chr(10))"
```

CLI `sqlite3` tidak terpasang di VPS ini — pakai `.venv/bin/python` seperti di atas.

Render satu halaman tanpa menyentuh produksi (contoh yang dipakai session ini):

```bash
.venv/bin/python typeset_test.py <page>/cleaned.png <page>/translations.json \
  --ocr-json <page>/ocr-result.json --reference-image <original> \
  --removal-report <page>/removal-report.json --report /tmp/out.json \
  --output /tmp/out.png --min-font-size 22 --min-scale-ratio .52 --layout-engine v5 \
  --surface-map <page>/surface-map.json --hard-clip
```

Banding visual (halaman acuan yang sudah benar): `data/jobs/.../chapter-1/<job_id>/metadata/page-005/`
→ `typeset.png`, `typeset-report.json`, `removal-report.json`, `mask-debug.png`, `hybrid-report.json`.

### Restart yang aman

```bash
pm2 restart ryukomik-translator-worker      # aman kalau antrean kosong
pm2 stop ryukomik-translator-worker && pm2 start ryukomik-translator-worker   # kalau mau drain dulu
```
Restart worker **di tengah halaman** memutus subprocess `translate.py`; halaman itu akan diulang dari
awal saat job di-resume, bukan dilanjutkan. Cek dulu tidak ada job `RUNNING`:
`sqlite3 data/translator.sqlite3 "select id,status,stage from jobs where status='RUNNING';"`.
Jangan restart `ryukomik-xianscan-ml` saat removal berjalan — sidecar mati membuat halaman itu
jatuh ke OpenCV dan hasil inpainting-nya berbeda (lihat §5.8).

## 8. TODO berikutnya (prioritas)

### 8.1 Aktif (masih menggantung)

1. **Normalisasi line ending** pada 7 file yang jadi CRLF (`source_typography.py`, `source_text_color.py`,
   `translate_text_test.py`, `memory_trainer.py`, `tests/test_source_typography.py`,
   `tests/test_source_text_color.py`, `tests/test_typeset_engine_v5.py`) supaya diff turun dari
   1904/1186 → 825/107. Ini dulu, sebelum commit.
2. Commit WIP per topik dengan path eksplisit (contoh grup: mask/typeset → branding → OCR resident →
   translate texts → reference memory → storage/ops → API). Jangan `git add -A`.
3. Perbaiki `ANCHOR` di `branding_v51.py` (§5.7) supaya `pytest` hijau — bug-nya sudah ada di HEAD.
4. 479 orphan OCR tanpa huruf berdiri sendiri — belum diputuskan.
5. 727 grup `source == translation` — perlu keputusan tuning.
6. Jalankan `python -m unittest` untuk 12 test baru + pastikan `tools/ops/storage_health.py` masuk cron
   kalau mau disk alert otomatis.

### 8.2 Dibatalkan / diterima apa adanya (JANGAN dijalankan)

- ~~Render ulang penuh job `7cfe2ac0` (atau minimal page-006..015) untuk memastikan mask growth tidak
  mengubah halaman lain~~ → **DIBATALKAN atas keputusan user** (2026-10-21). Job `7cfe2ac0` sudah
  dinyatakan **SELESAI** (§6), jadi pengukuran efek mask growth di halaman lain tidak diperlukan.
  Jangan dijalankan.
- ~~Sisa ~21 px teks masih agak ke atas di page-005 karena mask berhenti 42 px di atas garis; butuh
  cara mendeteksi kontur outline, bukan melonggarkan ambang~~ → **DITERIMA apa adanya**, dipindah ke
  §5.12. **Dibiarkan oleh user, jangan diubah tanpa perintah baru.**

## 9. File yang wajib dibuka dulu

Urutan baca yang paling cepat paham: `queue_worker.py` → `translate.py` → `translate_text_test.py` →
`remove_text_test.py` → `typeset_engine_v5.py`. Sisanya kontekstual:

- `translate_test.py` — OCR PaddleOCR → `ocr-result.json`. `resident_ocr.py` **[UNTRACKED]** — service OCR job-scoped (`chapter_ocr`), dipakai job >1 file.
- `translation_providers.py` — provider 9router/deepseek/gemini/openai-compatible. `memory_trainer.py` — istilah/register per chapter.
- `quality_pipeline_v2.py` — surface analysis & QC. `enclosed_bubble.py` — deteksi + growth mask bubble. `footprint_masks.py` **[UNTRACKED]** — mask footprint hemat memori.
- `hybrid_inpaint.py` / `xianscan_client.py` / `xianscan_sidecar.py` — klien + server sidecar inpaint.
- `typeset_test.py` — CLI typeset per halaman. `source_typography.py` / `source_text_color.py` — tiru font/italic dan warna sumber.
- `branding_v51.py` / `branding_v52.py` / `kencomics_mark.py` **[UNTRACKED]** — kredit sumber & branding gutter.
- `reference_memory.py` **[UNTRACKED]** / `memory_alignment.py` **[UNTRACKED]** / `pronoun_policy.py` **[UNTRACKED]** — referensi terjemahan & pronominal.
- `raw_join.py` **[UNTRACKED]** — gabung halaman lossless. `disk_budget.py` **[UNTRACKED]** — admission disk. `job_queue.py` / `queue_db.py` — enqueue (`--raw-mode`) & `claim_next`.
- `webp_archive.py` — ZIP/WebP chapter. `review_render.py` — render ulang saat edit bubble.
- `ecosystem.config.cjs` — semua env runtime. `api/src/server.ts` — REST dashboard.
- `tools/ops/storage_health.py` **[UNTRACKED]**, `tools/ops/maintain_storage.py` **[UNTRACKED]**, `tools/ops/retain_deploy_backups.py` **[UNTRACKED]** — health & retensi disk/backup.
- `tools/repair/rerender_job_typeset.py`, `tools/repair/requeue_job.py` — perbaikan job tanpa antre ulang manual.

**[UNTRACKED]** = belum masuk git; kalau hilang harus ditulis ulang dari nol. Inventaris lengkap 46 entri ada di §6.1.

## 10. Belum terverifikasi

- Klaim "sidecar XianScan bocor RSS ~42 MB/hari" berasal dari catatan session; belum diukur ulang di sini.
- Apakah `resident_ocr.py` benar-benar aktif di job produksi terakhir (butuh cek log `TRANSLATOR_OCR_SOCKET`).
  Yang terverifikasi hanya: default env `'1'` dan syarat `input > 1 file`.
- Efek mask growth `enclosed_bubble.py` pada halaman 006–015 tetap **tidak diukur** — render ulangnya
  dibatalkan atas keputusan user (§8.2). Jangan dijadikan blocker; kalau butuh angka, minta perintah baru.
- Perilaku akhir `reference_memory.py` / `memory_alignment.py` / `pronoun_policy.py` disimpulkan dari
  docstring + nama fungsi + test baru, belum diuji end-to-end ke job nyata.
- `vendor/xianscan` berubah sebagai submodule/isi vendor; isi diff-nya belum ditelusuri baris per baris.
- `models/lama.onnx` lokal kosong; belum dipastikan apakah ada job lama yang masih memakai mode `on` lokal.
