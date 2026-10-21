import unittest

import cv2
import numpy as np

from remove_text_test import flat_glyph_fill_mask


def leaf_region(x, y, width, height):
    return {"bbox": {"x": x, "y": y, "width": width, "height": height}}


class FlatGlyphFillTests(unittest.TestCase):
    """White-fill hanya untuk huruf; garis bubble tidak boleh ikut hilang.

    pyscho ch1 page-007: mask persegi milik baris terakhir menutup busur
    garis awan di bawahnya, lalu diisi putih rata sehingga tampak kotak clean
    yang menembus bubble.
    """

    @classmethod
    def setUpClass(cls):
        cls.image = np.full((220, 320, 3), 255, np.uint8)
        # tiga huruf tiruan
        for x in (60, 100, 140):
            cv2.rectangle(cls.image, (x, 40), (x + 24, 64), (0, 0, 0), -1)
        # garis pembungkus di dalam footprint: panjang, tipis, menembus tepi
        cv2.line(cls.image, (10, 80), (310, 80), (0, 0, 0), 4)
        cls.regions = [leaf_region(50, 30, 130, 40)]
        cls.permitted = np.zeros((220, 320), np.uint8)
        cls.permitted[24:90, 44:200] = 255  # footprint persegi, menutup garis
        cls.mask = flat_glyph_fill_mask(cls.image, cls.permitted, cls.regions)

    def ink(self):
        return self.image.min(axis=2) < 180

    def test_garis_bubble_tidak_ikut_terhapus(self):
        arc = np.zeros((220, 320), bool)
        arc[78:84, :] = True
        arc &= self.ink()
        self.assertGreater(int(np.count_nonzero(arc)), 200, "busur uji kosong")
        self.assertEqual(0, int(np.count_nonzero(arc & (self.mask > 0))),
                         "mask pembersih menutup garis bubble")

    def test_huruf_tetap_terhapus_semua(self):
        letters = np.zeros((220, 320), bool)
        letters[35:70, 50:175] = True
        letters &= self.ink()
        covered = np.count_nonzero(letters & (self.mask > 0))
        self.assertGreaterEqual(covered / np.count_nonzero(letters), 0.98)

    def test_mask_selalu_di_dalam_footprint(self):
        self.assertEqual(0, int(np.count_nonzero(self.mask & (self.permitted == 0))))

    def test_tanpa_garis_mask_tetap_penuh(self):
        clean = np.full((120, 160, 3), 255, np.uint8)
        for x in (30, 70, 110):
            cv2.rectangle(clean, (x, 40), (x + 22, 66), (0, 0, 0), -1)
        permitted = np.zeros((120, 160), np.uint8)
        permitted[30:80, 20:140] = 255
        mask = flat_glyph_fill_mask(clean, permitted, [leaf_region(25, 35, 115, 36)])
        self.assertGreater(int(np.count_nonzero(mask)), 3000)


if __name__ == "__main__":
    unittest.main()
