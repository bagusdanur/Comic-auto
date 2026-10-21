"""Test anchor cadangan di tengah area aman.

`fit_mask_center` lahir dari pyscho apartment ch1 page-005 bubble-002: mask
aman setinggi 268 px, tetapi anchor-nya diambil dari titik tengah glyph sumber
(y=258.5), jadi hanya ~10 px tersisa di bawah blok teks. `fit_text_v5` menyerah
di semua ukuran, jalur darurat juga gagal, dan halaman itu berakhir
`PRESERVED / UNFIT_SOURCE_RESTORED` -- teks Inggris asli dikembalikan dan
pembaca melihat satu gelembung yang tidak diterjemahkan.

Tengah bbox piksel aman selalu berada di dalam area yang bisa dipakai, jadi
fit diulang di sana sebelum menyerah. Titik asal glyph yang sudah muat tidak
pernah ikut bergeser: retry hanya jalan ketika `fitted` kosong.

Jalankan: PYTHONPATH=<root> .venv/bin/python tests/test_mask_center_anchor.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from typeset_engine_v5 import fit_mask_center, fit_text_v5  # noqa: E402

FONT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets/fonts/animeace/animeace2bb_tt/animeace2_bld.ttf")

# teks sepanjang panel denda di pyscho page-005
TEXT = "PENJARA MAKSIMAL LIMA TAHUN ATAU DENDA SEPULUH JUTA YEN"
MINIMUM, MAXIMUM = 22, 32


class MaskCenterAnchorTest(unittest.TestCase):
    def test_anchor_di_tepi_bawah_buntu_center_muat(self):
        mask = np.full((268, 400), 255, np.uint8)
        # titik tengah glyph sumber persis seperti mesin menghitungnya
        self.assertIsNone(
            fit_text_v5(TEXT, FONT, mask, MINIMUM, MAXIMUM, 0,
                        anchor_center=(200, 258.5)),
            "anchor 10 px dari dasar mask seharusnya tidak muat")
        fitted = fit_mask_center(TEXT, FONT, mask, MINIMUM, MAXIMUM, 0)
        self.assertIsNotNone(fitted, "tengah area aman harusnya muat")
        self.assertEqual(fitted["fallback_reason"], "MASK_CENTER_ANCHOR")
        self.assertGreaterEqual(fitted["font_size"], MINIMUM)

    def test_tengah_area_aman_bukan_tengah_array(self):
        # separuh kiri mask diblokir: tengah array (200) jatuh di batas,
        # tengah piksel aman ada di 300.
        mask = np.zeros((300, 400), np.uint8)
        mask[:, 200:] = 255
        self.assertIsNone(
            fit_text_v5(TEXT, FONT, mask, MINIMUM, MAXIMUM, 0,
                        anchor_center=(20, 20)))
        fitted = fit_mask_center(TEXT, FONT, mask, MINIMUM, MAXIMUM, 0)
        self.assertIsNotNone(fitted)
        self.assertGreaterEqual(fitted["origin"][0], 200,
                                "teks harus mendarat di area yang boleh dipakai")

    def test_mask_kosong_tidak_melempar(self):
        self.assertIsNone(
            fit_mask_center(TEXT, FONT, np.zeros((50, 50), np.uint8),
                            MINIMUM, MAXIMUM, 0))

    def test_jejak_fallback_ditandai(self):
        # telemetri butuh penanda ini supaya jumlah penyelamatan bisa dihitung
        mask = np.full((268, 400), 255, np.uint8)
        fitted = fit_mask_center(TEXT, FONT, mask, MINIMUM, MAXIMUM, 0)
        self.assertIn("MASK_CENTER_ANCHOR", str(fitted.get("fallback_reason")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
