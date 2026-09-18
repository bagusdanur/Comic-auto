#!/usr/bin/env python3
"""Resource-bounded LaMa ONNX inpainting for artwork text masks.

The model is optional: callers can keep the deterministic OpenCV path when the
weight or onnxruntime is unavailable. Only local patches around connected mask
clusters are inferred, preserving all pixels outside the supplied glyph mask.

The implementation follows the patch strategy used by XianScan (MIT) and the
LaMa ONNX model (Apache-2.0); it is adapted to this pipeline's strict pixel mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class HybridResult:
    image: np.ndarray
    backend: str
    patch_count: int
    changed_outside_mask: int


class LamaPatchInpainter:
    def __init__(self, model_path: Path, threads: int = 2, session=None) -> None:
        self.model_path = Path(model_path)
        if session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"LaMa model not found: {self.model_path}")
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError("onnxruntime is not installed") from error
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = max(1, min(2, int(threads)))
            options.enable_cpu_mem_arena = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"], sess_options=options
            )
        self.session = session
        inputs = session.get_inputs()
        if len(inputs) < 2:
            raise RuntimeError("LaMa model must expose image and mask inputs")
        image_input, mask_input = inputs[0], inputs[1]
        if len(image_input.shape) == 4 and image_input.shape[1] == 1:
            image_input, mask_input = mask_input, image_input
        self.image_name, self.mask_name = image_input.name, mask_input.name

    def _infer(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        pad_bottom = (8 - height % 8) % 8
        pad_right = (8 - width % 8) % 8
        if pad_bottom or pad_right:
            image_rgb = cv2.copyMakeBorder(
                image_rgb, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE
            )
            mask = cv2.copyMakeBorder(
                mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=0
            )
        image_tensor = image_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        mask_tensor = (mask > 0).astype(np.float32)[None, None]
        output = self.session.run(
            None, {self.image_name: image_tensor, self.mask_name: mask_tensor}
        )[0][0]
        if output.ndim == 3 and output.shape[0] in (1, 3, 4):
            output = output.transpose(1, 2, 0)
        if float(output.max()) <= 1.01:
            output = output * 255.0
        return np.clip(output, 0, 255).astype(np.uint8)[:height, :width, :3]

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray, pad: int = 32) -> HybridResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected an RGB image")
        if mask.shape != image_rgb.shape[:2]:
            raise ValueError("Mask dimensions differ from image")
        binary = np.where(mask > 0, 255, 0).astype(np.uint8)
        if not np.any(binary):
            return HybridResult(image_rgb.copy(), "lama-onnx-patch", 0, 0)

        # Join nearby letters into word/line patches so LaMa sees enough native
        # context, while final compositing still uses the exact glyph mask.
        cluster_seed = cv2.dilate(binary, np.ones((17, 17), np.uint8), iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cluster_seed, 8)
        result = image_rgb.copy()
        patch_count = 0
        height, width = binary.shape
        for label in range(1, count):
            x, y, box_width, box_height = (int(value) for value in stats[label, :4])
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(width, x + box_width + pad), min(height, y + box_height + pad)
            patch_mask = binary[y1:y2, x1:x2]
            if not np.any(patch_mask):
                continue
            inferred = self._infer(result[y1:y2, x1:x2], patch_mask)
            select = patch_mask > 0
            patch = result[y1:y2, x1:x2]
            patch[select] = inferred[select]
            result[y1:y2, x1:x2] = patch
            patch_count += 1

        changed = np.any(result != image_rgb, axis=2)
        outside = int(np.count_nonzero(changed & (binary == 0)))
        if outside:
            raise RuntimeError(f"LaMa pixel safety failed: {outside} pixels outside mask")
        return HybridResult(result, "lama-onnx-patch", patch_count, outside)


def create_lama(model_path: Path | None, threads: int = 2):
    if model_path is None or not Path(model_path).is_file():
        return None
    try:
        return LamaPatchInpainter(Path(model_path), threads=threads)
    except (ImportError, RuntimeError, OSError):
        return None
