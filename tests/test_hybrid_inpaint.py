import unittest

import cv2
import numpy as np

from hybrid_inpaint import LamaPatchInpainter


class _Input:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _FakeSession:
    def get_inputs(self):
        return [_Input("image", [1, 3, "h", "w"]), _Input("mask", [1, 1, "h", "w"])]

    def run(self, _, values):
        image = values["image"].copy()
        mask = values["mask"] > 0
        image[np.repeat(mask, 3, axis=1)] = 0.75
        return [image]


class HybridInpaintTests(unittest.TestCase):
    def test_lama_patch_composites_only_exact_glyph_mask(self):
        image = np.full((96, 180, 3), 80, dtype=np.uint8)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.putText(mask, "RAW", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 255, 3, cv2.LINE_AA)
        engine = LamaPatchInpainter("unused.onnx", session=_FakeSession())
        result = engine(image, mask)
        changed = np.any(result.image != image, axis=2)
        self.assertGreater(result.patch_count, 0)
        self.assertEqual(result.changed_outside_mask, 0)
        self.assertEqual(int(np.count_nonzero(changed & (mask == 0))), 0)
        self.assertGreater(int(np.count_nonzero(changed & (mask > 0))), 0)

    def test_empty_mask_is_pixel_identical(self):
        image = np.full((32, 32, 3), 123, dtype=np.uint8)
        engine = LamaPatchInpainter("unused.onnx", session=_FakeSession())
        result = engine(image, np.zeros((32, 32), dtype=np.uint8))
        self.assertTrue(np.array_equal(result.image, image))
        self.assertEqual(result.patch_count, 0)


if __name__ == "__main__":
    unittest.main()
