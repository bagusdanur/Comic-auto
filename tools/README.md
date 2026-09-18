# Operational tools

Operational scripts are grouped by purpose:

- `audit/`: read-only validation and consistency reports
- `review/`: contact sheets, crops, and visual previews
- `repair/`: targeted corrections and re-rendering
- `maintenance/`: migrations, archive rebuilding, and cleanup
- `ops/`: monitoring, smoke checks, model downloads, and diagnostics

Run tools from the project root so imports and relative runtime paths remain
stable, for example:

```bash
python -m tools.audit.audit_reviews_compact <job-directory>
python -m tools.ops.monitor_job <job-id>
```

Create a space-efficient regression baseline for a completed job without
copying its page images:

```bash
python -m tools.audit.build_pipeline_baseline <job-directory> \
  --output artifacts/baselines/<name>.json --hashes
```

Production entry points remain at the project root because PM2 and worker
subprocesses use those stable paths.
