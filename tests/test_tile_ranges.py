"""Test pembagian tile OCR.

Ekor tile yang lebih pendek dari setengah TILE_HEIGHT adalah lubang, bukan
tile.  Dengan TILE_HEIGHT=1600 dan OVERLAP=200, halaman setinggi 13000 px
berakhir dengan (12600,13000) -- 400 px -- dan detektor mengembalikan kosong
dari sana, padahal piksel yang sama terbaca 0.80-0.88 dari crop 1000 px.
Kotak 'FEAR' ch1 page-001 (y12815-12999) ada di ekor itu, jadi tidak pernah
terdeteksi: tak pernah diterjemahkan, tak pernah dihapus.

Jalankan: PYTHONPATH=<root> .venv/bin/python tests/test_tile_ranges.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translate_test import tile_ranges  # noqa: E402


def covers(ranges, height):
    """Tidak ada celah: setiap baris 0..height-1 masuk minimal satu tile.

    Minimal satu, bukan tepat satu -- tile memang sengaja bertumpuk sebesar
    TILE_OVERLAP, jadi menghitung kembaran sebagai celah adalah salah.
    """
    covered = bytearray(height)
    for a, b in ranges:
        for i in range(a, b):
            covered[i] += 1
    return all(c >= 1 for c in covered)


class TileTailTest(unittest.TestCase):
    """Ekor panjang-gagal harus dilipat ke tile sebelumnya."""

    def test_halaman_13000_tidak_menghasilkan_ekor_sliver(self):
        # sebelum: [(11200,12800),(12600,13000)] -> 400 px, detektor kosong
        ranges = tile_ranges(13000)
        self.assertGreaterEqual(ranges[-1][1] - ranges[-1][0], 1600 // 2,
                                f"ekor {ranges[-1]} terlalu pendek")

    def test_blok_fear_masih_tercakup_setelah_dilipat(self):
        ranges = tile_ranges(13000)
        pembawa = [t for t in ranges if t[0] <= 12815 < t[1]]
        self.assertEqual(len(pembawa), 1)
        top, bottom = pembawa[0]
        self.assertGreater(bottom - top, 1000,
                           "tile pembawa blok FEAR harus cukup tinggi")

    def test_lipatan_menutup_baris_yang_sama_tanpa_celah(self):
        for height in (13000, 11782, 12999, 8000, 4000, 1600, 900, 1):
            with self.subTest(height=height):
                ranges = tile_ranges(height)
                self.assertTrue(covers(ranges, height),
                                f"celah pada tinggi {height}: {ranges}")

    def test_baris_terakhir_selalu_termasuk(self):
        for height in (13000, 11782, 5001, 1601):
            with self.subTest(height=height):
                self.assertEqual(tile_ranges(height)[-1][1], height)

    def test_halaman_pendek_tidak_dilipat_salah(self):
        # di bawah setengah TILE_HEIGHT hanya ada satu tile: tak ada yang dilipat
        ranges = tile_ranges(900)
        self.assertEqual(ranges, [(0, 900)])

    def test_tile_penuh_tidak_dilipat(self):
        # tinggi kelipatan TILE_HEIGHT-OVERLAP berakhir pas: ekor = tile penuh
        for height in (1600, 3000, 4400, 5800):
            with self.subTest(height=height):
                ranges = tile_ranges(height)
                self.assertEqual(ranges[-1][1], height)
                self.assertEqual(ranges[-1][1] - ranges[-1][0],
                                 min(1600, height))


class TileOverlapAndValidationTest(unittest.TestCase):
    def test_tile_bertumpuk_sesuai_overlap(self):
        ranges = tile_ranges(13000)
        for (a, b), (c, d) in zip(ranges, ranges[1:]):
            if b < 13000:   # baris terakhir baru dilipat, jadi tumpang tindihnya beda
                self.assertEqual(a + 1600 - 200, c, "langkah harus TILE_HEIGHT-OVERLAP")

    def test_konfigurasi_tidak_valid_ditolak(self):
        import translate_test as T
        old_h, old_o = T.TILE_HEIGHT, T.TILE_OVERLAP
        try:
            T.TILE_HEIGHT, T.TILE_OVERLAP = 0, 0
            with self.assertRaises(ValueError):
                T.tile_ranges(1000)
            T.TILE_HEIGHT, T.TILE_OVERLAP = 1600, 1600
            with self.assertRaises(ValueError):
                T.tile_ranges(1000)
        finally:
            T.TILE_HEIGHT, T.TILE_OVERLAP = old_h, old_o


if __name__ == "__main__":
    unittest.main()
