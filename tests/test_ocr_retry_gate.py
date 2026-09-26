"""Test syarat penerimaan retry OCR.

Regresi: `retry` lama menuntut `joined_letters > region_letters`, jadi baris
yang sampahnya berhuruf banyak selalu ditolak meski pembacaan ulangnya benar
pada conf 0.98 -- "MOVING NEXT WEEK, RIGHT?" jadi 'TXENSNIAOE' (10 huruf) dan
'SLRIIETEEE' (10), padahal bacaan benarnya 'MOVING NEXT' (10) dan
'WEEK, RIGHT?' (9).  Keduanya gugur dan bahasa Inggris lolos ke halaman
selesai.  Sisi lain yang harus tetap terjaga: kasus ch6, di mana bacaan benar
memang lebih panjang dari sampahnya ('ANLIAE SE E' -> "HEY, DON'T KEEP
US WAITING"), dan jangan sampai crop sempit menciutkan baris jadi token.

Jalankan: PYTHONPATH=<root> .venv/bin/python tests/test_ocr_retry_gate.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translate_test import retry_better  # noqa: E402


def letters(text):
    return len("".join(c for c in text.lower() if c.isalpha()))


class RetryGateRegressionTest(unittest.TestCase):
    """Bug ch1 Archmage: bacaan benar berhuruf sama/sedikit dari sampah."""

    def test_moving_next_diterima_meski_jumlah_huruf_sama(self):
        # 'TXENSNIAOE' (10) vs 'MOVING NEXT' (10): syarat lama 10>10 = False.
        self.assertTrue(retry_better(letters("MOVING NEXT"), 0.98,
                                     letters("TXENSNIAOE"), 0.25))

    def test_week_right_diterima_meski_huruf_lebih_sedikit(self):
        # 'SLRIIETEEE' (10) vs 'WEEK, RIGHT?' (9): syarat lama 9>10 = False.
        self.assertTrue(retry_better(letters("WEEK, RIGHT?"), 0.97,
                                     letters("SLRIIETEEE"), 0.16))

    def test_kasus_ch6_yang_lebih_panjang_tetap_diterima(self):
        # Syarat lama memang dibuat untuk ini; jangan sampai kembali rusak.
        self.assertTrue(retry_better(letters("HEY, DON'T KEEP US WAITING"), 0.95,
                                     letters("ANLIAE SE E"), 0.28))


class RetryGateGuardsTest(unittest.TestCase):
    """Yang harus tetap ditolak."""

    def test_crop_sempit_dilarang_menciutkan_baris_jadi_token(self):
        # crop yang cuma kena 'OK' conf 1.0 tidak boleh menggantikan 10 huruf
        self.assertFalse(retry_better(letters("OK"), 1.0,
                                      letters("TXENSNIAOE"), 0.25))

    def test_tidak_menerima_skor_sama_atau_lebih_rendah(self):
        region_letters = letters("MOVING NEXT WEEK")
        self.assertFalse(retry_better(letters("MOVING NEXT"), 0.25,
                                      region_letters, 0.25))
        self.assertFalse(retry_better(letters("MOVING NEXT"), 0.20,
                                      region_letters, 0.25))

    def test_lantai_huruf_minimum_tiga(self):
        # region asli pendek: tetap butuh minimal 3 huruf
        self.assertFalse(retry_better(letters("OK"), 0.99, 2, 0.10))
        self.assertTrue(retry_better(letters("YES"), 0.99, 2, 0.10))

    def test_paruh_region_menerima_persis_paruh(self):
        # batas bawah: >= setengah panjang region asli diterima
        region_letters = 20
        self.assertTrue(retry_better(10, 0.9, region_letters, 0.3))
        self.assertFalse(retry_better(9, 0.9, region_letters, 0.3))


if __name__ == "__main__":
    unittest.main()
