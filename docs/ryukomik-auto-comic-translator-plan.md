# PROJECT: RYUKOMIK AUTO COMIC TRANSLATOR

## FINAL QUALITY UPGRADE PLAN — 30 AGUSTUS 2026

Status: **SAFE WITH LIMITATIONS**. Bagian ini menjadi rencana implementasi final
dan menggantikan strategi kualitas lama yang mengandalkan perluasan bbox serta
aturan khusus per judul. Ketentuan keamanan dan dokumentasi historis di bawah
tetap berlaku.

### Kapasitas VPS live

- CPU: 4 vCPU; load saat audit `1.42 / 0.89 / 0.72`.
- RAM: 7,8 GiB total; 3,9 GiB available; swap terpakai 495 MiB dari 2 GiB.
- Disk: 27 GiB tersedia dari 58 GiB.
- Service terbesar saat audit: dua proses Node sekitar 844 MB dan 819 MB RSS.
- Worker translator idle sekitar 170 MB RSS dan dibatasi PM2 1,5 GB.
- Kesimpulan: satu gambar aktif aman; dua gambar paralel tidak dipakai secara default.

### Sasaran kualitas dan waktu

- Halaman sederhana: waktu tambahan QC maksimal 10–20% dari pipeline sekarang.
- Halaman kompleks: maksimal dua render deterministik; AI vision hanya untuk
  metadata/layout yang ambigu, bukan untuk menggambar pixel halaman.
- Target otomatis bersih: >=97% region; target review: <=3% region.
- Klaim 100% hanya jika seluruh pemeriksaan otomatis lulus. Region gagal tidak
  boleh dipaksa menjadi `COMPLETED`.
- Dimensi output harus identik; pixel di luar edit mask tidak boleh berubah.

### Arsitektur final: fast path + selective escalation

```text
RAW original
  -> tiled OCR + polygon/baseline
  -> surface association
  -> contextual translation
  -> background-safe cleanup
  -> constrained typesetting
  -> cheap residual/collision QC
       PASS -> output
       FAIL -> deterministic retry sekali
       FAIL ambigu -> AI layout metadata saja
       FAIL artwork -> REVIEW
```

AI tidak dijalankan pada semua bubble. OpenCV/Pillow menangani geometri dan
pixel; provider LLM hanya membantu pengelompokan/klasifikasi ketika confidence
rendah. Tidak ada image generation pada fast path.

### Stage 1 — OCR dan geometri RAW

Simpan untuk setiap baris:

- polygon empat titik, baseline angle, confidence, tinggi glyph;
- warna glyph, warna/tekstur sekitar, dan reading order;
- tile-local serta koordinat global.

Gabungkan baris hanya jika berada pada surface yang sama. Kedekatan bbox saja
tidak cukup. Dedup overlap tile memakai polygon IoU dan kemiripan teks.

Biaya: tidak menambah inference model; hanya post-processing OpenCV ringan.

### Stage 2 — surface association

Klasifikasi deterministik:

- `BUBBLE_FLAT`, `BUBBLE_COLORED`, `CAPTION_RECT`, `TILTED_CARD`;
- `CHAT_UI`, `ARTWORK_TEXT`, `SFX`, `LOGO_WATERMARK`.

Gunakan contour/edge/warna lokal. Setiap surface memiliki polygon luar dan
safe polygon hasil inset. Jika satu OCR group memotong dua surface, grup wajib
dipecah. Jika dua grup mengklaim surface sama secara tidak masuk akal, tandai
`SURFACE_AMBIGUOUS`.

AI vision hanya dipanggil untuk `SURFACE_AMBIGUOUS`, dibatch per halaman dan
hanya mengembalikan JSON metadata—tidak mengedit gambar.

### Stage 3 — cleanup berdasarkan kondisi

- Flat putih/warna solid: hapus mask glyph dan isi dari median interior.
- Gradient ringan: interpolasi warna dari sisi mask.
- Kotak RAW yang memang opaque: pertahankan polygon/ukuran/sudut asli dan isi
  warna permukaan yang terdeteksi; jangan memperbesar kotak.
- Artwork: inpaint hanya mask glyph dengan dilation kecil.
- Artwork tidak aman: jangan diedit; masuk review.

Final rendering selalu dimulai dari `original`, bukan output render sebelumnya.
Mask maksimum tidak boleh melewati safe polygon surface.

### Stage 4 — constrained typesetting

- Font awal mengikuti median tinggi glyph RAW.
- Style mengikuti RAW: dialogue, narration, thought, shout, chat, atau SFX.
- Warna/stroke diambil dari RAW dan contrast permukaan.
- Teks miring dirender pada bidang rectified, kemudian dikembalikan memakai
  transformasi RAW; clipping keras memakai safe polygon.
- Line wrapping memilih baris paling seimbang dan menghindari satu kata per baris.
- Bubble/kotak tidak pernah diperbesar.
- Jika tidak muat pada minimum readable size, minta terjemahan lebih ringkas
  sekali; jika masih gagal, `TEXT_OVERFLOW_REVIEW`.

### Stage 5 — QC murah pada semua region

QC wajib dan CPU ringan:

1. ukuran gambar sama;
2. perubahan pixel berada di allowed mask;
3. alpha/text mask berada 100% di safe polygon;
4. tidak ada tabrakan text mask;
5. ukuran font dan jumlah baris masih wajar dibanding RAW;
6. edge/noise score pada hasil cleanup tidak melonjak;
7. OCR residual hanya pada crop yang diedit, bukan OCR ulang seluruh halaman.

Residual OCR crop dijalankan hanya bila detector murah masih melihat kandidat
stroke. Dengan demikian halaman bersih tidak membayar OCR kedua penuh.

### Retry dan review policy

- Retry 1: kecilkan font/ubah wrap atau tambah dilation mask maksimum 1–2 px.
- Retry 2 hanya untuk metadata ambiguous dan memakai AI layout batch halaman.
- Tidak ada retry ketiga.
- Artwork reconstruction, font di bawah batas baca, residual RAW, collision,
  atau surface confidence rendah masuk review dengan nomor halaman dan alasan.
- Status `COMPLETED` hanya jika seluruh hard checks lulus.

### Resource guard final

```env
TRANSLATOR_CONCURRENCY=1
TRANSLATOR_MAX_CPU_THREADS=2
TRANSLATOR_MIN_FREE_RAM_MB=2500
TRANSLATOR_PAUSE_LOAD_1M=3.20
TRANSLATOR_TILE_HEIGHT=2000
TRANSLATOR_TILE_OVERLAP=200
QC_MAX_DETERMINISTIC_RETRIES=1
QC_AI_ESCALATION_MAX_PER_PAGE=1
QC_RESIDUAL_CROP_ONLY=true
```

- Worker menunggu bila available RAM <2,5 GiB atau load 1 menit >3,2.
- Peak RSS worker ditargetkan <=1,2 GiB; PM2 hard limit tetap 1,5 GiB.
- Decode/proses satu gambar; release buffer sebelum gambar berikutnya.
- Cache translation/font metrics boleh dipakai; bitmap full page tidak dicache.

### Implementasi bertahap tanpa menghentikan produksi

1. Tambahkan feature flag `QUALITY_PIPELINE_V2=false` dan kumpulkan benchmark
   baseline dari 10 halaman representatif.
2. Implementasikan data model polygon/surface dan unit test tanpa mengubah output.
3. Implementasikan cleanup dispatch serta render dari original.
4. Implementasikan hard clipping, balanced wrapping, dan collision test.
5. Implementasikan residual-crop QC serta alasan review per halaman.
6. Shadow-run V2 pada 10–20 halaman; output lama tetap production.
7. Bandingkan waktu, RAM, residual RAW, overflow, dan kerusakan pixel.
8. Aktifkan V2 untuk job baru jika acceptance gate lulus. Job lama tidak diubah.
9. Pertahankan `QUALITY_PIPELINE_V2=false` sebagai rollback satu langkah selama
   minimal tujuh hari.

### Acceptance gate

- Peak available RAM tidak turun di bawah 2 GiB.
- Worker RSS <=1,2 GiB dan tidak menyebabkan restart service lain.
- Median waktu per halaman naik <=20%; p95 naik <=35%.
- Zero dimension mismatch dan zero changed pixel di luar allowed mask.
- Zero typeset overflow/collision pada region berstatus selesai.
- Residual RAW pada region selesai <=1%; sisanya otomatis review.
- Review memiliki indikator halaman dan alasan yang dapat ditindaklanjuti.

### Urutan pengerjaan yang disetujui oleh plan ini

Mulai dari Stage 1–2 dan hard clipping terlebih dahulu. AI escalation serta
residual OCR adaptif dipasang terakhir setelah fast path stabil. Dengan urutan
ini kualitas dasar membaik tanpa langsung menambah waktu dan biaya pada semua
halaman.

### Implementation record — V2 shadow

Implemented on 30 August 2026:

- `QUALITY_PIPELINE_V2=off|shadow|on` feature flag;
- deterministic `surface-map.json` with RAW polygon, outer/safe polygon,
  angle, colour statistics, confidence, collision, and ambiguity metadata;
- cleanup-mask clipping against the detected surface;
- hard typeset clipping, collision protection, and `SURFACE_CLIP_LOSS` guard;
- `quality-report.json` with dimension, outside-mask, outside-surface,
  component diagnostics, and cheap RAW residual checks;
- worker RAM/load guard and CPU thread limits;
- isolated unit test and shadow rendering; production output remains V1.

Measured on a 690x12661 difficult page:

- surface analysis: 3.51 s, about 288 MB peak RSS;
- hard validation: initially 1.81 s; with residual checks about 3.43 s;
- hard-clipped V2: zero pixels outside allowed mask and zero rendered pixels
  outside detected surfaces;
- two regions correctly rejected for clipping loss;
- four residual candidates detected, so the acceptance gate has not yet passed.

Current deployment state: `shadow`. Do not switch to `on` until residual false
positives are calibrated on the 10-page benchmark set and ambiguous surface
rate is at or below the acceptance target.

Saya ingin membuat aplikasi internal untuk menerjemahkan komik/webtoon
dari Bahasa Inggris ke Bahasa Indonesia secara otomatis.

## PRINSIP UTAMA

Aplikasi ini akan berjalan di VPS yang SUDAH menjalankan banyak project/service.

Karena itu:

1. JANGAN langsung install dependency.
2. JANGAN langsung membuat aplikasi.
3. JANGAN langsung menjalankan OCR/model.
4. WAJIB audit kondisi VPS terlebih dahulu.
5. Setelah audit, tentukan apakah aplikasi ini aman dijalankan.
6. Jika tidak aman, STOP dan berikan rekomendasi.
7. Jangan mengganggu service Ryukomik yang sudah berjalan.

Aplikasi translator harus:
- CPU only
- tidak membutuhkan GPU
- hemat RAM
- concurrency processing = 1
- tidak menjalankan LLM lokal
- translation menggunakan external API
- tidak menggunakan image generation
- tidak mengubah artwork komik
- tidak mengubah resolusi original

---

## PHASE 0 — AUDIT VPS TERLEBIH DAHULU

INI WAJIB DILAKUKAN SEBELUM CODING ATAU INSTALL APAPUN.

Periksa VPS saat ini secara LIVE.

Kumpulkan:

### CPU

Jalankan dan analisis:

```bash
lscpu
nproc
uptime
top
```

Gunakan `htop` jika tersedia.

Catat:
- jumlah CPU/core
- load average
- CPU idle
- process CPU terbesar

### RAM

Jalankan:

```bash
free -h
cat /proc/meminfo
```

Catat:
- total RAM
- used
- available
- cache
- swap usage

Jangan hanya melihat "free RAM".

Gunakan nilai AVAILABLE RAM sebagai pertimbangan utama.

### SWAP

Jalankan:

```bash
swapon --show
free -h
```

Periksa apakah VPS sudah sering menggunakan swap.

Jika swap sangat tinggi, cari tahu apakah karena memory pressure lama
atau memang RAM saat ini sempit.

### DISK

Jalankan:

```bash
df -h
```

Gunakan `du -sh` pada direktori project utama jika relevan.

Periksa:
- total storage
- storage available
- filesystem tempat translator akan menyimpan temp image

Translator akan mengolah file gambar besar,
jadi pastikan disk temporary cukup.

### PROCESS

Periksa:

```bash
ps aux --sort=-%mem | head -30
ps aux --sort=-%cpu | head -30
```

Jika PM2 tersedia:

```bash
pm2 list
pm2 monit
pm2 jlist
```

Identifikasi service:
- Ryukomik frontend
- Ryukomik backend
- bot
- database
- Hermes
- service lainnya

JANGAN stop atau restart service apapun selama audit.

### DATABASE

Periksa apakah ada:
- PostgreSQL
- MySQL
- SQLite service lain

Translator sendiri TIDAK membutuhkan PostgreSQL.

### NODE

Periksa:

```bash
node -v
npm -v
```

### PYTHON

Periksa:

```bash
python3 --version
pip3 --version
```

Jangan mengganti versi Python global jika service lain bergantung padanya.

Gunakan virtual environment khusus translator.

### NGINX

Periksa:

```bash
nginx -v
systemctl status nginx
```

Jangan mengubah config production sebelum plan disetujui.

---

## PHASE 0 — RESOURCE FEASIBILITY REPORT

Setelah audit, STOP.

Jangan install apapun dulu.

Berikan report seperti:

```text
RYUKOMIK TRANSLATOR VPS FEASIBILITY

CPU
- Core:
- Current load:
- Available headroom:

RAM
- Total:
- Used:
- Available:
- Swap:

DISK
- Total:
- Available:

CURRENT SERVICES
- service A:
- service B:
- dst

ESTIMATED TRANSLATOR RESOURCE

Idle:
- RAM:
- CPU:

During OCR:
- estimated peak RAM:
- CPU threads:
- temporary disk:

RISK LEVEL:
LOW / MEDIUM / HIGH

VERDICT:

SAFE TO INSTALL
atau

NOT RECOMMENDED
```

Berikan alasan.

---

## RESOURCE REQUIREMENT

Target architecture translator:

Idle translator:
target <= 200-300 MB RAM

Ketika OCR/processing aktif:
target tambahan RAM <= 1.0-1.5 GB

CPU processing:
maksimal 2 CPU thread secara default

Concurrent image processing:
1

Jangan menjalankan dua OCR job secara bersamaan.

Jika RAM available sebelum processing kurang dari:

```text
2000 MB
```

job harus menunggu.

Config:

```env
TRANSLATOR_CONCURRENCY=1
TRANSLATOR_MIN_FREE_RAM_MB=2000
TRANSLATOR_MAX_CPU_THREADS=2
```

Jika audit menunjukkan resource tidak mencukupi,
jangan lanjutkan deployment.

---

## SETELAH SAYA MENYETUJUI PHASE 0

Baru lanjut ke implementasi berikut.

---

## ARCHITECTURE

Gunakan architecture ringan:

```text
Browser
   |
   v
Vite SPA
   |
   v
Fastify API
   |
   v
Job Queue
   |
   v
Python Image Worker
   |
   +---- Text Detection
   |
   +---- OCR
   |
   +---- Translation API
   |
   +---- Text Cleaning
   |
   +---- Indonesian Typesetting
   |
   v
Output Image / ZIP
```

---

## FRONTEND

Gunakan:

- Vite
- React
- TypeScript
- Tailwind CSS

JANGAN gunakan Next.js.

Alasan:
translator adalah internal dashboard SPA dan tidak membutuhkan:
- SSR
- SEO
- Server Components

Build:

```bash
npm run build
```

Production frontend diserve langsung sebagai static files melalui Nginx.

Jangan menjalankan Vite dev server di production.

Target:
- ringan
- fast
- smooth
- mobile friendly
- desktop friendly

Jangan install UI framework besar jika tidak diperlukan.

Gunakan dependency seminimal mungkin.

---

## FRONTEND PERFORMANCE

Jangan menampilkan original 690x16000 secara langsung di dashboard list.

Backend harus menghasilkan preview kecil / thumbnail.

Original full-resolution hanya digunakan saat:
- OCR
- final rendering
- detailed review bila diperlukan

Lazy-load halaman editor/review.

Gunakan dynamic import.

Jangan decode puluhan full-resolution webtoon secara bersamaan di browser.

---

## BACKEND API

Gunakan:

- Node.js
- Fastify
- TypeScript

Alasan memilih Fastify:
- ringan
- cepat
- cocok untuk API internal
- tidak membutuhkan framework fullstack berat

Backend menangani:

- authentication sederhana jika diperlukan
- upload
- job management
- queue
- communication dengan Python worker
- status
- download
- settings

---

## PYTHON WORKER

Gunakan Python worker terpisah.

Python worker hanya menangani image/OCR.

Gunakan virtual environment:

```text
translator-worker/.venv
```

JANGAN install dependency Python secara global.

Worker stack:

- Pillow
- OpenCV
- NumPy
- PaddleOCR lightweight/mobile
- optional comic text detector

Semua model harus CPU compatible.

---

## OCR STRATEGY

Prioritas utama:
hemat resource.

Gunakan lightweight/mobile English OCR model.

Jangan langsung menggunakan model OCR server/large.

Sebelum menetapkan model final:
benchmark minimal satu sample comic.

Ukur:
- OCR accuracy
- processing time
- peak RAM

Jika PaddleOCR versi yang terinstall menggunakan nama/model berbeda,
gunakan English mobile model yang kompatibel dengan versi tersebut.

Jangan hardcode nama model tanpa mengecek dokumentasi/version yang terinstall.

---

## TEXT DETECTION

Evaluasi:

`comic-text-detector`

Repository:

https://github.com/dmMaze/comic-text-detector

Tetapi JANGAN otomatis menggunakannya jika terlalu berat untuk VPS.

Benchmark terlebih dahulu.

Jika dependency detector terlalu berat,
gunakan pendekatan lebih ringan:

```text
OCR detector bawaan PaddleOCR
+
filtering
+
OpenCV
```

Prioritas:

```text
RESOURCE SAFETY > kompleksitas model
```

Tujuannya adalah menemukan:
- text bounding box
- text polygon
- confidence
- reading order

---

## IMAGE RULE — SANGAT PENTING

JANGAN gunakan:
- Stable Diffusion
- DALL-E
- image generation
- generative full image editing
- image reconstruction seluruh halaman

Artwork original TIDAK BOLEH dibuat ulang.

Contoh:

```text
INPUT:
690 x 16000

OUTPUT:
690 x 16000
```

WAJIB:

```text
output.width == input.width
output.height == input.height
```

Jika berbeda bahkan 1 pixel:
job gagal.

Tidak boleh:
- resize
- crop
- padding
- white bar
- canvas expansion
- aspect ratio change

---

## PIXEL PRESERVATION

Tujuan utama:

pixel di luar text-editing region harus sama dengan original
sebisa mungkin.

Workflow:

```text
original image
        |
        +---- OCR/detection
        |
        +---- text mask
        |
        +---- edit hanya masked area
        |
        +---- render translation
        |
        v
output
```

Jangan menjalankan enhancement/filter/color correction
ke seluruh image.

Tidak boleh:
- sharpening seluruh image
- blur seluruh image
- recomposition
- resize
- global denoise

---

## LONG WEBTOON SUPPORT

Support image seperti:

```text
690x5000
690x10000
690x16000
690x20000
690x30000
```

Long images harus diproses sebagai vertical tiles.

Contoh default:

```env
TRANSLATOR_TILE_HEIGHT=2000
TRANSLATOR_TILE_OVERLAP=200
```

Contoh:

```text
image:
690x16000

OCR chunks:

0-2200
2000-4200
4000-6200
dst.
```

Overlap digunakan supaya text di perbatasan tidak terpotong.

Setelah detection:

ubah local tile coordinate menjadi global image coordinate.

Lakukan duplicate suppression pada overlap.

Tile hanya untuk inference.

JANGAN menyimpan output dalam bentuk tile.

Final output tetap satu original canvas.

---

## MEMORY HANDLING

Jangan load seluruh chapter sekaligus.

Untuk batch:

```text
001.webp
002.webp
003.webp
004.webp
```

Process:

```text
001
release memory

002
release memory

003
release memory
```

Setelah image selesai:

- close Pillow objects
- release NumPy arrays
- release OpenCV matrices
- release temporary buffers

Gunakan:

```python
del object
```

dan jika perlu:

```python
gc.collect()
```

Jangan melakukan `gc.collect()` berlebihan jika tidak dibutuhkan.

---

## THREAD LIMIT

Batasi CPU worker.

Target default:
2 threads.

Atur jika supported:

- OpenCV thread limit
- OMP_NUM_THREADS
- MKL_NUM_THREADS
- Paddle thread configuration

Jangan biarkan OCR mengambil seluruh 4 CPU core terus menerus.

---

## OCR OUTPUT

Setiap detected text memiliki metadata:

```json
{
  "id": "region-id",
  "pageId": "...",
  "text": "What are you doing?",
  "confidence": 0.97,
  "bbox": {
    "x": 100,
    "y": 250,
    "width": 220,
    "height": 90
  },
  "polygon": [],
  "type": "dialogue",
  "status": "OK"
}
```

---

## TEXT CLASSIFICATION

Bedakan sebisa mungkin:

- dialogue
- narration
- SFX
- decorative text

Default:

```text
translate_dialogue = true
translate_narration = true
translate_sfx = false
```

Jangan menghapus:
- Japanese/Korean SFX
- watermark
- logo
- credit
- artwork decorative text

kecuali user memang mengaktifkan fitur tersebut.

---

## OCR CONFIDENCE

Default:

```env
OCR_CONFIDENCE_THRESHOLD=0.75
```

Jika confidence rendah:
jangan asal hapus tulisan.

Set:

```text
status = LOW_OCR_CONFIDENCE
```

dan masukkan ke review.

---

## TRANSLATION

JANGAN menjalankan LLM di VPS.

Gunakan external API.

Buat provider abstraction:

```text
TranslationProvider
```

Implement minimal:

```text
GeminiProvider
DeepSeekProvider
```

Optional:

```text
OpenAICompatibleProvider
```

Environment:

```env
TRANSLATION_PROVIDER=
TRANSLATION_MODEL=
TRANSLATION_API_KEY=
```

API key hanya backend.

JANGAN expose API key ke Vite/browser.

---

## TRANSLATION CONTEXT

Jangan translate satu bubble dengan satu API request.

Urutkan text:

```text
top -> bottom
```

Kumpulkan beberapa text dalam satu batch.

Contoh request:

```json
[
  {
    "id": "r1",
    "text": "What are you doing?"
  },
  {
    "id": "r2",
    "text": "Leave me alone."
  }
]
```

System instruction translator:

```text
You are a professional English to Indonesian comic translator.

Translate dialogue naturally into Indonesian.

Do not translate word-for-word if it sounds unnatural.

Preserve:
- emotion
- character personality
- jokes
- insults
- tone
- context

Keep translations concise enough for comic speech bubbles.

Do not add explanation.

Do not censor ordinary dialogue.

Return only valid structured JSON.

Never change region IDs.
```

Output:

```json
[
  {
    "id": "r1",
    "translation": "Apa yang sedang kau lakukan?"
  },
  {
    "id": "r2",
    "translation": "Biarkan aku sendiri."
  }
]
```

Validate JSON.

Retry API response if malformed.

---

## GLOSSARY

Implement glossary per project.

Examples:

- character name
- skill name
- organization
- location
- special terms

Example:

```json
{
  "Black Flame": "Api Hitam",
  "Mana Core": "Inti Mana"
}
```

Translation prompt harus mempertahankan glossary.

---

## TEXT REMOVAL

Ini area paling berisiko.

Prioritaskan bubble putih/flat color.

Untuk text hitam di speech bubble putih:

gunakan:

1. OCR/text mask
2. mask dilation kecil
3. background sampling
4. local fill / OpenCV
5. hanya edit area huruf

Jangan menghapus seluruh bubble.

Jangan menghancurkan bubble border.

---

## COMPLEX BACKGROUND

Jika tulisan berada langsung di artwork/background kompleks:

JANGAN langsung menggunakan model generative besar.

Default:

```env
COMPLEX_INPAINT=false
```

Jika text removal tidak aman:

```text
COMPLEX_BACKGROUND
NEEDS_REVIEW
```

Lebih baik meminta review manual
daripada merusak artwork.

---

## TYPESETTING

Gunakan Pillow.

Implement auto-fit font.

Workflow:

1. ambil target region
2. tentukan max text area
3. tentukan initial font size
4. word wrap
5. hitung bounding box
6. jika overflow, kurangi font
7. ulangi sampai fit

Support:

- multiline
- center
- left
- right
- line spacing
- configurable font
- stroke
- letter spacing jika memungkinkan

Default dialogue:
center aligned.

---

## FONT FIT

Jangan biarkan translation keluar dari bubble.

Config:

```env
TYPESET_MIN_FONT_SIZE=
TYPESET_MAX_FONT_SIZE=
```

Jika translation tidak dapat dimasukkan ke bubble
bahkan pada minimum size:

```text
TEXT_TOO_LARGE
NEEDS_REVIEW
```

Jangan otomatis memperbesar bubble.

---

## ORIGINAL IMAGE SAFETY

Simpan original.

Jangan pernah overwrite original.

Storage:

```text
data/
    jobs/
        {job-id}/
            original/
            output/
            preview/
            metadata/
            temp/
```

Final rendering selalu dimulai dari ORIGINAL.

Jika user mengedit translation:
jangan edit output berkali-kali.

Render ulang region dari original.

Tujuannya menghindari degradation berulang.

---

## OUTPUT FORMAT

Support:

- WebP
- PNG
- JPG/JPEG

Default:
preserve original format.

Jika input:

```text
.webp
```

output:

```text
.webp
```

Jika input:

```text
.png
```

output:

```text
.png
```

---

## WEBP QUALITY

Jangan otomatis convert WebP ke JPEG.

Gunakan high-quality WebP encoding.

Quality configurable.

Contoh:

```env
WEBP_QUALITY=95
```

Tetapi ukuran pixel harus tetap sama.

---

## BATCH PROCESSING

Support:

- single image
- multiple images
- ZIP chapter

ZIP example:

```text
001.webp
002.webp
003.webp
004.webp
```

Output harus mempertahankan nama dan urutan.

Jangan rename random.

---

## QUEUE

Gunakan lightweight persistent queue.

JANGAN gunakan:
- Redis
- RabbitMQ
- Kafka
- Celery

untuk versi pertama.

Gunakan:
- SQLite
atau
- lightweight file/job database

Job status:

```text
QUEUED
WAITING_RESOURCE
PROCESSING
REVIEW
COMPLETED
FAILED
CANCELLED
```

---

## RESOURCE GUARD

Sebelum menjalankan OCR:

cek available RAM.

Jika:

```text
available RAM < TRANSLATOR_MIN_FREE_RAM_MB
```

jangan start processing.

Status:

```text
WAITING_RESOURCE
```

Cek lagi beberapa waktu kemudian.

Jangan crash service lain hanya untuk translator.

---

## DATABASE

Gunakan SQLite.

Data:

```text
jobs
images
regions
settings
glossaries
```

Tidak perlu PostgreSQL untuk translator.

---

## WEB UI

Design:
dark modern Ryukomik style.

Prioritaskan usability, bukan animasi berat.

### DASHBOARD

Tampilkan:
- upload
- worker status
- queue
- RAM available
- CPU load
- jobs terbaru

### UPLOAD

Support:
- drag & drop
- image
- multiple image
- ZIP

### JOB DETAIL

Tampilkan:
- jumlah gambar
- progress
- OCR progress
- translation progress
- processing
- completed
- review count
- failed

### REVIEW

Tampilkan preview comic.

Overlay bounding box untuk region.

User dapat klik region.

Editor:

```text
Original:
What are you doing?

Translation:
Apa yang sedang kau lakukan?
```

Controls:

- translation text
- font size
- X
- Y
- width
- height
- alignment
- line spacing
- re-render

Jika user mengubah satu bubble:
jangan OCR ulang semua gambar.

Hanya render ulang region yang diperlukan.

---

## PREVIEW

Jangan kirim original 690x16000 ke dashboard hanya untuk thumbnail.

Generate preview terpisah.

Preview boleh resized.

FINAL OUTPUT TIDAK BOLEH resized.

---

## API

Minimal endpoint:

```text
POST /api/jobs

GET /api/jobs

GET /api/jobs/:jobId

POST /api/jobs/:jobId/cancel

GET /api/jobs/:jobId/images

GET /api/images/:imageId/regions

PATCH /api/regions/:regionId

POST /api/images/:imageId/render

GET /api/jobs/:jobId/download

GET /api/system/status

GET /api/settings

PATCH /api/settings
```

---

## SYSTEM STATUS API

Return:

```json
{
  "cpu": {
    "usage": 0
  },
  "memory": {
    "total": 0,
    "available": 0
  },
  "worker": {
    "status": "idle"
  },
  "queue": {
    "waiting": 3
  }
}
```

Jangan update UI setiap 100ms.

Polling ringan misalnya:
3-5 detik.

---

## UPLOAD SECURITY

Validate:

- MIME
- extension
- image integrity
- file size

Protect ZIP extraction dari:

zip-slip / path traversal.

Jangan percaya filename user.

Sanitize filename.

---

## TEMP CLEANUP

Temporary files jangan menumpuk.

Implement cleanup.

Contoh:

hapus temp file setelah job selesai.

Optional:
hapus completed output setelah configurable retention period.

Jangan hapus original/output tanpa policy yang jelas.

---

## NGINX

Frontend production:

Vite build:

```text
dist/
```

Nginx serve static dist.

API:

```text
/translator-api/
```

reverse proxy ke Fastify.

Jangan mengganggu existing Ryukomik Nginx hosts.

Gunakan subdomain terpisah jika memungkinkan:

```text
translator.domain.tld
```

atau internal path yang aman.

Jangan ubah config yang sudah berjalan tanpa backup.

Sebelum reload:

```bash
nginx -t
```

Hanya reload jika test berhasil.

---

## PROCESS MANAGEMENT

Fastify API dapat dijalankan dengan PM2.

Python worker juga dapat dijalankan sebagai process terpisah.

Tetapi OCR model jangan terus melakukan inference saat idle.

Pastikan worker idle menggunakan resource sekecil mungkin.

Jika lebih hemat:
model OCR boleh lazy-load saat job pertama.

Pertimbangkan unload model setelah idle timeout
jika memory VPS membutuhkan.

---

## LOGGING

Log:

- job started
- job completed
- job failed
- OCR duration
- translation duration
- render duration
- peak memory jika bisa
- error

Jangan log:
- API key
- secret
- full sensitive request header

---

## TEST FILE

Gunakan beberapa kategori test:

1. short comic
2. 690x16000 webtoon
3. speech bubble putih
4. narration box
5. text dekat batas tile
6. text background kompleks
7. WebP
8. PNG
9. JPEG

---

## PIXEL SAFETY TEST

Implement test otomatis.

Sebelum dan sesudah:

```text
assert dimensions equal
```

Selain mask region:

bandingkan original dan output.

Pixel modification di luar allowed edit mask
harus seminimal mungkin.

Buat debug mode yang menghasilkan:

```text
mask-debug.png
```

showing area yang diizinkan untuk berubah.

---

## BENCHMARK

Setelah prototype berhasil:

benchmark 1 gambar sekitar 690x16000.

Report:

- processing time
- peak RAM
- average CPU
- OCR accuracy estimate
- number of text regions
- translation duration
- output size

Kemudian test batch kecil.

Jangan langsung test 100 gambar.

---

## IMPLEMENTATION PHASES

### PHASE 0

VPS AUDIT.

STOP dan laporkan hasil.

Tunggu approval saya.

### PHASE 1

Buat Python OCR prototype.

Command:

```bash
python translate_test.py input.webp
```

Hanya:
- open
- tile
- OCR
- bounding boxes
- debug output

Belum translate.

Benchmark RAM.

STOP dan laporkan.

### PHASE 2

Translation provider.

Tambahkan Gemini/DeepSeek.

Belum edit image.

Print:

```text
English -> Indonesian
```

Test translation context.

### PHASE 3

Text masking/removal.

Prioritaskan bubble putih.

Test apakah artwork rusak.

### PHASE 4

Typesetting Indonesia.

Implement auto-fit.

Pastikan dimensions tetap sama.

### PHASE 5

Full CLI pipeline.

Example:

```bash
python translate.py input.webp output.webp
```

### PHASE 6

SQLite + Job queue.

Concurrency = 1.

Resource guard.

### PHASE 7

Fastify API.

### PHASE 8

Vite React TypeScript UI.

### PHASE 9

Review editor.

### PHASE 10

ZIP/batch support.

### PHASE 11

Production deployment.

---

## IMPORTANT DEVELOPMENT RULE

JANGAN membuat semua phase sekaligus.

Setelah setiap phase:

1. test
2. benchmark
3. tunjukkan hasil
4. jelaskan resource usage
5. pastikan tidak merusak existing service

Baru lanjut.

---

## FAIL-SAFE

Jika Python OCR membutuhkan RAM terlalu besar:

STOP.

Jangan memaksakan deployment.

Evaluasi:
- model lebih kecil
- OCR alternatif lebih ringan
- tile lebih kecil
- worker on-demand
- OCR external API

Jika comic-text-detector terlalu berat:

hapus dari architecture.

Gunakan PaddleOCR detector saja.

Jika translation API gagal:
jangan menghapus English text.

Region harus tetap original.

Jika OCR gagal:
jangan edit region.

Jika typesetting gagal:
jangan merusak original.

---

## TARGET RESULT

Saya ingin workflow akhirnya:

```text
Upload chapter ZIP

          ↓

Queue

          ↓

OCR English

          ↓

Translate Indonesia

          ↓

Hapus text English HANYA di area yang aman

          ↓

Typeset Indonesia

          ↓

Review bubble bermasalah

          ↓

Download ZIP
```

Contoh:

```text
INPUT:
chapter.zip

001.webp = 690x16000
002.webp = 690x14500
003.webp = 690x18000

OUTPUT:

001.webp = 690x16000
002.webp = 690x14500
003.webp = 690x18000
```

Tidak boleh blur seluruh halaman.
Tidak boleh ada white bar.
Tidak boleh resize.
Tidak boleh menggambar ulang karakter.
Tidak boleh mengubah panel.

---

## PRIORITY ORDER

Urutan prioritas:

1. Jangan mengganggu VPS production.
2. Jangan merusak artwork.
3. Pertahankan resolusi asli.
4. RAM usage rendah.
5. OCR accuracy.
6. Translation quality.
7. Typesetting quality.
8. Processing speed.

Lebih baik processing agak lambat
daripada membuat VPS production kehabisan RAM.

---

## MULAI SEKARANG

Untuk sekarang HANYA lakukan PHASE 0.

Audit VPS.

JANGAN install dependency translator.
JANGAN clone repository.
JANGAN npm install.
JANGAN pip install.
JANGAN mengubah Nginx.
JANGAN restart PM2.
JANGAN restart server.

Setelah audit:

berikan FEASIBILITY REPORT lengkap dan rekomendasi:

```text
SAFE
SAFE WITH LIMITATIONS
atau
NOT SAFE
```

Kemudian STOP dan tunggu instruksi saya.
