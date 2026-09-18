import unittest

import cv2
import numpy as np

from remove_text_test import (
    artwork_text_mask, flat_bubble_residual_mask, hybrid_artwork_eligible, locally_flat_text_surface,
    padded_group_box, reconstruct_flat_surface, region_mask,
)


class ArtworkRemovalTests(unittest.TestCase):
    def test_padded_group_box_unions_lines_and_stays_in_safe_surface(self):
        regions = [
            {"bbox": {"x": 40, "y": 35, "width": 80, "height": 18}},
            {"bbox": {"x": 55, "y": 58, "width": 95, "height": 18}},
        ]
        surface = {"safe_polygon": [[35, 25], [160, 25], [160, 85], [35, 85]]}
        box = padded_group_box(regions, (120, 220, 3), surface)
        self.assertLessEqual(box[0], 40)
        self.assertLessEqual(box[1], 35)
        self.assertGreaterEqual(box[2], 150)
        self.assertGreaterEqual(box[3], 76)
        self.assertGreaterEqual(box[0], 35)
        self.assertGreaterEqual(box[1], 25)

    def test_local_flat_surface_accepts_white_bubble_with_dark_letters(self):
        image = np.full((80, 180, 3), 252, dtype=np.uint8)
        cv2.putText(image, "TEXT", (48, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (8, 8, 8), 2)
        self.assertTrue(locally_flat_text_surface(image, (30, 15, 150, 65)))

    def test_residual_mask_removes_inner_letters_but_not_bubble_outline(self):
        image = np.full((120, 240, 3), 252, dtype=np.uint8)
        cv2.ellipse(image, (120, 60), (105, 48), 0, 0, 360, (8, 8, 8), 3)
        cv2.putText(image, "RAW", (78, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (8, 8, 8), 3)
        surface = {"safe_polygon": [[22, 24], [218, 24], [218, 96], [22, 96]]}
        mask = flat_bubble_residual_mask(image, surface)
        self.assertGreater(np.count_nonzero(mask[45:78, 70:175]), 100)
        self.assertEqual(np.count_nonzero(mask[:, :18]), 0)

    def test_hybrid_rejects_credit_and_dense_masks(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:50, 20:80] = 255
        surface = {"surface_type": "ARTWORK_TEXT"}
        self.assertFalse(hybrid_artwork_eligible(True, surface, [{"mask_ratio": 0.2}], mask))
        self.assertFalse(hybrid_artwork_eligible(False, surface, [{"mask_ratio": 0.6}], mask))
        self.assertTrue(hybrid_artwork_eligible(False, surface, [{"mask_ratio": 0.2}], mask))

    def test_flat_surface_reconstruction_keeps_local_gradient(self):
        image = np.zeros((100, 180, 3), dtype=np.uint8)
        for x in range(image.shape[1]):
            image[:, x] = (235 + x // 18, 238 + x // 22, 245)
        image[42:58, 65:115] = 20
        restored, mask = reconstruct_flat_surface(image, (60, 35, 120, 65))
        self.assertEqual(int(np.count_nonzero(mask)), 60 * 30)
        self.assertGreater(float(restored[50, 105].mean()), 225)
        self.assertLess(abs(float(restored[50, 62].mean()) - float(restored[50, 118].mean())), 12)

    def test_flat_bubble_mask_does_not_cross_bbox_edge_lines(self):
        image = np.full((100, 220, 3), 255, dtype=np.uint8)
        bbox = {"x": 30, "y": 20, "width": 160, "height": 60}
        cv2.line(image, (30, 24), (189, 30), (25, 25, 25), 2, cv2.LINE_AA)
        cv2.putText(image, "HELP", (72, 61), cv2.FONT_HERSHEY_SIMPLEX, .8, (10, 10, 10), 2, cv2.LINE_AA)
        mask, details = region_mask(image, bbox)
        self.assertIsNotNone(mask, details)
        assert mask is not None
        crop = mask[20:80, 30:190]
        self.assertEqual(np.count_nonzero(crop[:, :2]), 0)
        self.assertEqual(np.count_nonzero(crop[:, -2:]), 0)

    def test_glyph_mask_and_inpaint_never_change_outside_region(self):
        height, width = 120, 260
        x_gradient = np.linspace(40, 190, width, dtype=np.uint8)
        image = np.repeat(x_gradient[None, :, None], height, axis=0)
        image = np.repeat(image, 3, axis=2)
        cv2.putText(image, "WORDS", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (10, 10, 10), 5, cv2.LINE_AA)
        cv2.putText(image, "WORDS", (45, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (245, 245, 245), 2, cv2.LINE_AA)
        bbox = {"x": 35, "y": 35, "width": 180, "height": 55}
        mask, details = artwork_text_mask(image, bbox)
        self.assertIsNotNone(mask, details)
        assert mask is not None
        self.assertEqual(details["reason"], "ARTWORK_GLYPH_INPAINT")
        self.assertGreater(np.count_nonzero(mask), 0)
        allowed_box = np.zeros((height, width), dtype=bool)
        allowed_box[35:90, 35:215] = True
        self.assertEqual(np.count_nonzero((mask > 0) & ~allowed_box), 0)
        result = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        changed = np.any(result != image, axis=2)
        self.assertEqual(np.count_nonzero(changed & (mask == 0)), 0)


if __name__ == "__main__":
    unittest.main()
