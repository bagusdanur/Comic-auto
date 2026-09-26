"""Test penskalaan mask typeset.

`fit_mask_size` lahir dari crash di Shadow Sovereign ch1 page-002:
`paint_fitted_text` memutar kanvas tegak (219,102) dengan `expand=True`, dan
Pillow membulatkan bounding box-nya menjadi (103,220) -- kelebihan 1px di
tiap sumbu.  Mask itu lalu dibandingkan dengan page-space guard setinggi
(size[1], size[0]) dan jatuh sebagai
`operands could not be broadcast together with shapes (220,103) (219,102)`.
Kanvas mendatar (102,219) tidak kena, jadi hanya caption tegak yang retak.

Jalankan: PYTHONPATH=<root> .venv/bin/python tests/test_fit_mask_size.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from typeset_engine_v5 import fit_mask_size, symmetric_outline_mask  # noqa: E402

# bounds yang benar-benar gagal di Shadow Sovereign ch1 page-002
SIZE = (102, 219)
GUARD = (SIZE[1], SIZE[0])          # bentuk yang diminta page-space guard


def transposed_canvas(size=SIZE):
    return Image.new("L", (size[1], size[0]))


class RotateThenFitTest(unittest.TestCase):
    """Reproduksi crash: kanvas tegak diputar lalu dipaskan."""

    def test_kanvas_tegak_diputar_90_tidak_lagi_bentrok(self):
        for rotation in (90, -90, 89.5, 90.5, 91, -89.5):
            with self.subTest(rotation=rotation):
                canvas = transposed_canvas()
                rotated = canvas.rotate(
                    -rotation, resample=Image.Resampling.BICUBIC,
                    expand=True, center=(SIZE[1] / 2, SIZE[0] / 2))
                # inilah ukuran liar yang menyebabkan crash
                self.assertNotEqual(rotated.size, SIZE)
                fitted = fit_mask_size(rotated, SIZE)
                self.assertEqual(fitted.size, SIZE)
                outline = np.asarray(symmetric_outline_mask(fitted, 3))
                self.assertEqual(outline.shape, GUARD,
                                 "outline_mask harus sama dengan guard")

    def test_kanvas_mendatar_tetap_sama(self):
        flat = Image.new("L", SIZE)
        rotated = flat.rotate(-12, resample=Image.Resampling.BICUBIC, expand=False)
        self.assertEqual(fit_mask_size(rotated, SIZE).size, SIZE)

    def test_semua_sudut_menghasilkan_bentuk_guard(self):
        canvas = transposed_canvas()
        for rotation in range(-180, 181, 15):
            with self.subTest(rotation=rotation):
                rotated = canvas.rotate(
                    -rotation, resample=Image.Resampling.BICUBIC,
                    expand=True, center=(SIZE[1] / 2, SIZE[0] / 2))
                fitted = fit_mask_size(rotated, SIZE)
                self.assertEqual(np.asarray(symmetric_outline_mask(fitted, 2)).shape,
                                 GUARD)


class ContentPreservationTest(unittest.TestCase):
    """Yang dipotong hanya pembulatan bounding box, bukan tinta."""

    def test_piksel_berwarna_tidak_berkurang(self):
        canvas = transposed_canvas()
        ImageDraw.Draw(canvas).rectangle((40, 30, 180, 90), fill=255)
        rotated = canvas.rotate(-90, resample=Image.Resampling.BICUBIC,
                                expand=True, center=(SIZE[1] / 2, SIZE[0] / 2))
        before = np.count_nonzero(np.asarray(rotated))
        after = np.count_nonzero(np.asarray(fit_mask_size(rotated, SIZE)))
        self.assertGreater(before, 0, "fixture harus punya tinta")
        self.assertEqual(after, before, "tinta tidak boleh hilang")

    def test_potongan_terpusat(self):
        mask = Image.new("L", (SIZE[0] + 2, SIZE[1] + 2), 0)
        ImageDraw.Draw(mask).point((1, 1), fill=255)          # di sudut surplus
        fitted = fit_mask_size(mask, SIZE)
        arr = np.asarray(fitted)
        # sudut surplus dipotong, isi tetap ada
        self.assertEqual(fitted.size, SIZE)
        self.assertEqual(np.count_nonzero(arr),
                         np.count_nonzero(np.asarray(
                             mask.crop((1, 1, 1 + SIZE[0], 1 + SIZE[1])))))


class EdgeCasesTest(unittest.TestCase):
    def test_mask_yang_sudah_tepat_dikembalikan_apa_adanya(self):
        same = Image.new("L", SIZE, 7)
        self.assertIs(fit_mask_size(same, SIZE), same)

    def test_mask_kekurangan_dipadding_dengan_tinta_kosong(self):
        small = Image.new("L", (100, 210), 128)
        out = fit_mask_size(small, SIZE)
        self.assertEqual(out.size, SIZE)
        arr = np.asarray(out)
        # penempatan terpusat: (102-100)//2 = 1 dan (219-210)//2 = 4
        # baris di luar mask asli harus tinta kosong, isinya tetap 128
        self.assertTrue(np.all(arr[0] == 0), "baris padatan atas harus kosong")
        self.assertTrue(np.all(arr[-1] == 0), "baris padatan bawah harus kosong")
        self.assertEqual(int(arr.max()), 128, "isi mask asli harus utuh")
        self.assertEqual(int(arr[4 + 100, 1 + 50]), 128)

    def test_bentuk_normal_bukan_persegi(self):
        for size in ((64, 64), (300, 90), (40, 400), (219, 102)):
            with self.subTest(size=size):
                canvas = Image.new("L", (size[1], size[0]))
                rotated = canvas.rotate(-90, resample=Image.Resampling.BICUBIC,
                                        expand=True,
                                        center=(size[1] / 2, size[0] / 2))
                fitted = fit_mask_size(rotated, size)
                self.assertEqual(fitted.size, size)
                self.assertEqual(np.asarray(symmetric_outline_mask(fitted, 1)).shape,
                                 (size[1], size[0]))


if __name__ == "__main__":
    unittest.main()
