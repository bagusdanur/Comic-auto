"""Production wrapper around the pinned XianScan FastAPI application.

It adds a resource-bounded detector-only endpoint.  The upstream analyze route
runs ComicTextDetector and RapidOCR together, which is too expensive for long
webtoon pages on the four-core VPS.  Existing upstream routes remain intact.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import Response


ROOT = Path(__file__).resolve().parent
XIANSCAN_ML = ROOT / "vendor" / "xianscan" / "ml"
if str(XIANSCAN_ML) not in sys.path:
    sys.path.insert(0, str(XIANSCAN_ML))

from app import main as upstream_main  # noqa: E402
from app import pipeline  # noqa: E402
from app.inpaint import get_inpainter  # noqa: E402


app = upstream_main.app


@app.post("/pages/detect")
def detect_only(image: UploadFile = File(...), include_mask: bool = Form(False)) -> dict:
    """Run ComicTextDetector without the full-page RapidOCR pass."""
    data = image.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > upstream_main.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        decoded = pipeline.decode_image(data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if pipeline.detector is None or not pipeline.detector.available():
        raise HTTPException(status_code=503, detail="ComicTextDetector unavailable")

    result = pipeline.detector.analyze(decoded)
    height, width = decoded.shape[:2]
    regions = []
    for index, box in enumerate(result.boxes):
        region = pipeline._region_from_box(box, index, width, height)
        if index < len(result.scores):
            region.confidence = max(0.0, min(1.0, float(result.scores[index])))
        regions.append(region.model_dump())

    response = {
        "width": width, "height": height, "backend": result.backend,
        "regions": regions, "mask_encoding": None, "mask_png_base64": None,
    }
    if include_mask and result.mask is not None:
        mask = result.mask
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != "uint8":
            maximum = float(mask.max()) if mask.size else 0.0
            scale = 255.0 if maximum <= 1.01 else 1.0
            mask = (mask * scale).clip(0, 255).astype("uint8")
        ok, encoded = cv2.imencode(".png", mask)
        if not ok:
            raise HTTPException(status_code=500, detail="could not encode detector mask")
        response["mask_encoding"] = "png-base64"
        response["mask_png_base64"] = base64.b64encode(encoded.tobytes()).decode("ascii")
    return response


@app.post("/pages/inpaint-mask")
def inpaint_exact_mask(image: UploadFile = File(...), mask: UploadFile = File(...),
                       mode: str = Form("patch")) -> Response:
    """Inpaint an explicit glyph mask and return a dimension-identical PNG."""
    image_data, mask_data = image.file.read(), mask.file.read()
    if not image_data or not mask_data:
        raise HTTPException(status_code=400, detail="image and mask are required")
    if len(image_data) > upstream_main.MAX_UPLOAD_BYTES or len(mask_data) > upstream_main.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")
    try:
        decoded = pipeline.decode_image(image_data)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    decoded_mask = cv2.imdecode(np.frombuffer(mask_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if decoded_mask is None:
        raise HTTPException(status_code=400, detail="invalid mask")
    if decoded_mask.shape != decoded.shape[:2]:
        raise HTTPException(status_code=400, detail="mask dimensions differ from image")
    binary = np.where(decoded_mask > 0, 255, 0).astype(np.uint8)
    ratio = float(np.mean(binary > 0))
    if ratio <= 0.0:
        raise HTTPException(status_code=400, detail="mask is empty")
    if ratio > 0.20:
        raise HTTPException(status_code=400, detail="mask exceeds page safety limit")
    try:
        result = get_inpainter()(decoded, binary, mode=mode)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    # Enforce the same exact-mask compositor used by the translator. LaMa may
    # infer a full patch, but only requested glyph pixels reach the response.
    composited = decoded.copy()
    selected = binary > 0
    composited[selected] = result[selected]
    ok, encoded = cv2.imencode(".png", composited)
    if not ok:
        raise HTTPException(status_code=500, detail="could not encode inpaint result")
    return Response(content=encoded.tobytes(), media_type="image/png")
