"""Regresi fill solid warna seragam untuk balon flat (anti-blur LaMa)."""
import unittest

import cv2
import numpy as np

from verified_flat_cleanup import verified_white_fill, verified_uniform_fill


def _bubble(color, textured=False):
    img = np.full((160, 220, 3), color, np.uint8)
    if textured:
        noise = np.random.RandomState(0).randint(0, 90, img.shape).astype(np.uint8)
        img = np.clip(img.astype(int) - noise, 0, 255).astype(np.uint8)
    # huruf gelap sparse di dalam bbox teks
    for x in (65, 80, 95):
        cv2.rectangle(img, (x, 65), (x + 7, 85), (10, 10, 10), -1)
    regions = [{"bbox": {"x": 60, "y": 60, "width": 80, "height": 30}}]
    return img, regions


class UniformFlatFillTests(unittest.TestCase):
    def test_off_white_seragam_diterima_white_fill_ditolak(self):
        img, regions = _bubble([240, 240, 238])
        surface = {"surface_type": "BUBBLE_FLAT"}
        self.assertIsNone(verified_white_fill(img, regions, surface))
        result = verified_uniform_fill(img, regions, surface)
        self.assertIsNotNone(result)
        mask, fill, verification = result
        self.assertLess(abs(int(fill[0]) - 240), 6)
        self.assertGreater(verification["ring_uniform_fraction"], 0.97)

    def test_balon_berwarna_seragam_diterima(self):
        img, regions = _bubble([205, 220, 240])
        surface = {"surface_type": "BUBBLE_COLORED"}
        result = verified_uniform_fill(img, regions, surface)
        self.assertIsNotNone(result)
        _, fill, _ = result
        self.assertGreater(fill[2], fill[0])  # biru > merah

    def test_artwork_bertekstur_ditolak(self):
        img, regions = _bubble([240, 240, 238], textured=True)
        surface = {"surface_type": "BUBBLE_FLAT"}
        self.assertIsNone(verified_uniform_fill(img, regions, surface))

    def test_artwork_text_surface_ditolak(self):
        img, regions = _bubble([240, 240, 238])
        self.assertIsNone(verified_uniform_fill(img, regions, {"surface_type": "ARTWORK_TEXT"}))

    def test_putih_murni_tetap_lewat_white_fill(self):
        img, regions = _bubble([255, 255, 255])
        surface = {"surface_type": "BUBBLE_FLAT"}
        self.assertIsNotNone(verified_white_fill(img, regions, surface))


if __name__ == "__main__":
    unittest.main()
