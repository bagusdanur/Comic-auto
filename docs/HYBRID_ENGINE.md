# Hybrid artwork cleanup

The production pipeline keeps its existing OCR, Indonesian translation memory,
bubble geometry, typesetting, review, WebP and ZIP stages. The optional hybrid
stage changes only cleanup of sparse source lettering over complex artwork.

Modes:

- `off`: existing cleanup only.
- `shadow`: classify and report LaMa eligibility without running inference.
- `on`: use local patch LaMa only after the safety gate passes; otherwise use
  the existing deterministic fallback.
- `xianscan-mask`: send the locally refined glyph mask to the shared XianScan
  LaMa sidecar. The response is composited through that exact mask and rejected
  if any pixel outside it changes. This is the resource-safe v3 candidate.

The safety gate rejects promotional/credit text, dense masks, non-artwork
surfaces and ambiguous geometry. The final compositor copies generated pixels
through the exact glyph mask, and aborts if any pixel outside that mask changes.

VPS defaults are intentionally `shadow`, one page at a time, two ONNX threads.
Enable `on` only after visual comparison of representative chapter pages.

Model/code provenance:

- Patch strategy adapted from XianScan (MIT):
  https://github.com/ArbenApura/xianscan
- Dynamic LaMa manga ONNX model (Apache-2.0 upstream attribution):
  https://huggingface.co/ogkalu/lama-manga-onnx-dynamic
