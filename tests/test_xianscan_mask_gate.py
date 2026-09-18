import unittest

import numpy as np

from remove_text_test import hybrid_artwork_eligible


class XianScanMaskGateTests(unittest.TestCase):
    def test_sparse_artwork_mask_is_eligible(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        surface = {"surface_type": "ARTWORK_TEXT"}
        self.assertTrue(hybrid_artwork_eligible(False, surface, [{"mask_ratio": 0.2}], mask))

    def test_dense_logo_mask_is_rejected(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[5:95, 5:95] = 255
        surface = {"surface_type": "ARTWORK_TEXT"}
        self.assertFalse(hybrid_artwork_eligible(False, surface, [{"mask_ratio": 0.829}], mask))


if __name__ == "__main__":
    unittest.main()
