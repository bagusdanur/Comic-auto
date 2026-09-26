const { readFileSync } = require('node:fs');
const root = '/home/ryukomik/ryukomik-auto-comic-translator';
const apiToken = readFileSync(`${root}/.api-token`, 'utf8').trim();

module.exports = {
  apps: [
    {
      name: 'ryukomik-xianscan-ml',
      cwd: `${root}/vendor/xianscan/ml`,
      script: `${root}/vendor/xianscan/ml/.venv/bin/uvicorn`,
      args: `--app-dir ${root} xianscan_sidecar:app --host 127.0.0.1 --port 8123`,
      interpreter: 'none',
      autorestart: true,
      max_memory_restart: '1800M',
      kill_timeout: 15000,
      env: {
        MT_DEVICE: 'cpu',
        MT_MODELS_DIR: `${root}/vendor/xianscan/ml/models`,
        OMP_NUM_THREADS: '2',
        OPENBLAS_NUM_THREADS: '2',
      },
    },
    {
      name: 'ryukomik-translator-api',
      cwd: root,
      script: `${root}/api/dist/start.js`,
      interpreter: 'node',
      autorestart: true,
      max_memory_restart: '180M',
      env: {
        NODE_ENV: 'production',
        TRANSLATOR_API_HOST: '127.0.0.1',
        TRANSLATOR_API_PORT: '3210',
        TRANSLATOR_API_TOKEN: apiToken,
        TRANSLATOR_PYTHON: `${root}/.venv/bin/python`,
        TRANSLATOR_DB: `${root}/data/translator.sqlite3`,
        TRANSLATOR_JOBS_DIR: `${root}/data/jobs`,
      },
    },
    {
      name: 'ryukomik-translator-worker',
      cwd: root,
      script: `${root}/queue_worker.py`,
      interpreter: `${root}/.venv/bin/python`,
      autorestart: true,
      // Measured 1600 MB peak on a single 60 000 px webtoon page (chapter 2
      // is five of them): full-page RGBA composites for every replaced logo
      // plus a full-page emblem scan. Both are now local and banded, but the
      // floor is still ~1.6 GB, so the old 1500M ceiling killed the worker
      // mid-branch and PM2 restarted it 222 times in a row.
      max_memory_restart: '2400M',
      kill_timeout: 15000,
      env: {
        PYTHONUNBUFFERED: '1',
        // Version gate only. "v5" is the current production pipeline and is
        // also what the translator API assigns to every new job; keeping the
        // worker default in sync prevents a job whose options are missing the
        // engine key from silently falling back to an older layout.
        PIPELINE_ENGINE: 'v5',
        TYPESET_LAYOUT_ENGINE: 'v5',
        QUALITY_PIPELINE_V2: 'on',
        HYBRID_PIPELINE: 'xianscan-v3',
        LAMA_MODEL_PATH: `${root}/models/lama.onnx`,
        XIANSCAN_ML_URL: 'http://127.0.0.1:8123',
        // XianScan keeps its ONNX detector/inpainter resident (~1 GB).  The
        // worker remains safe at 2.2 GB free while avoiding false aborts.
        TRANSLATOR_MIN_FREE_RAM_MB: '2200',
        TRANSLATOR_PAUSE_LOAD_1M: '3.20',
        OMP_NUM_THREADS: '2',
        OPENBLAS_NUM_THREADS: '2',
        MKL_NUM_THREADS: '2',
      },
    },
    {
      name: 'ryukomik-memory-worker',
      cwd: root,
      script: `${root}/memory_trainer.py`,
      interpreter: `${root}/.venv/bin/python`,
      autorestart: true,
      max_memory_restart: '1500M',
      kill_timeout: 15000,
      env: {
        PYTHONUNBUFFERED: '1',
        TRANSLATOR_MIN_FREE_RAM_MB: '2000',
      },
    },
  ],
};
