# Project structure

- `api/`: translator HTTP API
- `dashboard/`: review dashboard
- `assets/`: fonts and branding assets
- `tests/`: automated Python tests
- `tools/`: audit, repair, review, migration, and deployment utilities
- `artifacts/`: disposable local test and visual-review outputs
- `docs/`: architecture and operational documentation
- `data/`: runtime database, jobs, previews, and archives
- `output/`: legacy generated output retained for compatibility

The root Python files are currently stable runtime entry points. They will be
moved behind compatibility wrappers only after imports, subprocess calls, and
the PM2 configuration have been migrated and tested.

