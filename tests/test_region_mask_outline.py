import unittest

import cv2
import numpy as np

from remove_text_test import region_mask


class RegionMaskOutlineTests(unittest.TestCase):
    """Mask per-kotak tidak boleh menghapus busur garis bubble.

    Kotak OCR yang rapat ke baris terakhir ikut memuat potongan busur garis
    awan. Dulu potongan itu lolos sebagai "glyph" (lebar 127 px, tinggi 22 px
    -- belum cukup memanjang untuk aturan garis), lalu diisi putih sehingga
    garis bubble tampak terpotong kotak.
    """

    @classmethod
    def setUpClass(cls):
        cls.image = np.full((90, 260, 3), 255, np.uint8)
        for x in (20, 50, 80, 110, 140):
            cv2.rectangle(cls.image, (x, 30), (x + 18, 58), (25, 25, 25), -1)
        # potongan busur: lebar, tipis, lebih rendah dari tinggi huruf
        cv2.ellipse(cls.image, (130, 95), (110, 30), 0, 200, 340, (25, 25, 25), 4)
        cls.bbox = {"x": 12, "y": 22, "width": 236, "height": 46}
        cls.mask, cls.stats = region_mask(cls.image, cls.bbox)

    def test_busur_tidak_ikut_dibersihkan(self):
        self.assertIsNotNone(self.mask, self.stats)
        band = self.mask[62:90, :]
        self.assertEqual(0, int(np.count_nonzero(band)), "busur garis ikut dihapus")

    def test_huruf_tetap_dibersihkan(self):
        self.assertEqual("SAFE_FLAT_BACKGROUND", self.stats["reason"])
        letters = self.mask[30:58, 20:158]
        self.assertGreater(int(np.count_nonzero(letters)), 0.9 * 5 * 18 * 28 * 0.55)


if __name__ == "__main__":
    unittest.main()
