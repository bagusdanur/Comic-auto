"""Test filter glyph-besar vs teks narasi.

ch1 page-004 menumpuk "AND" (conf 1.00, 130x106) tepat di atas
"WHAT FOLLOWED NEXT WAS-" dalam kotak hitam yang sama.  Baris 13-huruf itu
lolos ukuran dan ke-TL, sedangkan kata sambung di atasnya ditolak sebagai
glyph raksasa -- tidak pernah digabung, tidak pernah dihapus, jadi bahasa
Inggris mentah tersisa di halaman selesai.  Sisi yang wajib terjaga: SFX aksi
('HHH', 'jHA') dan kata biasa berukuran besar ('CAT') tetap ditolak.

Jalankan: PYTHONPATH=<root> .venv/bin/python tests/test_narration_glyph_filter.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from translate_text_test import group_dialogue  # noqa: E402


def region(text, confidence=1.0, width=130, height=106, x=279, y=1642):
    return {
        "id": f"r-{text}-{width}x{height}",
        "text": text,
        "confidence": confidence,
        "bbox": {"x": x, "y": y, "width": width, "height": height},
        "type": "dialogue",
        "status": "OK",
    }


def passes(regions):
    return bool(group_dialogue(regions, 0))


class NarrationConnectiveTest(unittest.TestCase):
    """Kata narasi tunggal di kotak besar harus diterima."""

    def test_and_dari_ch1_p004_lolos(self):
        # region asli, koordinat persis seperti di ocr-result.json
        real = {"id": "x", "text": "AND", "confidence": 1.0,
                "bbox": {"x": 279, "y": 1642, "width": 130, "height": 106},
                "type": "dialogue", "status": "OK"}
        self.assertTrue(passes([real]))

    def test_setiap_kata_sambung_diterima_di_kotak_besar(self):
        for word in ("and", "but", "or", "so", "yet", "then", "now", "why"):
            with self.subTest(word=word):
                self.assertTrue(passes([region(word.upper())]),
                                f"{word} seharusnya diterima")

    def test_kata_sambung_conf_rendah_tetap_ditolak(self):
        # lolos glyph, tapi tetap kena lantai kepercayaan trustworthy_region
        self.assertFalse(passes([region("AND", confidence=0.50)]))


class GlyphStillRejectedTest(unittest.TestCase):
    """Yang ditolak aturan glyph harus tetap ditolak."""

    def test_sfx_aksi_besar_masih_ditolak(self):
        self.assertFalse(passes([region("HHH", confidence=0.32,
                                        width=334, height=743)]))
        self.assertFalse(passes([region("jHA", confidence=0.51,
                                        width=158, height=1070)]))

    def test_kata_biasa_besar_keyakinan_rendah_tetap_ditolak(self):
        # glyph aslinya dibuat untuk ini: bacaan yakin di bawah 0.85
        self.assertFalse(passes([region("CAT", confidence=0.60)]))
        self.assertFalse(passes([region("ZOO", confidence=0.70)]))

    def test_kata_biasa_besar_keyakinan_tinggi_kini_diterima(self):
        # perubahan disengaja: conf>=0.85 berarti teks nyata, bukan lettering.
        # 'Cat' conf 1.00 di seluruh repo termasuk yang salah ditolak sebelumnya.
        self.assertTrue(passes([region("CAT", confidence=1.00)]))
        self.assertTrue(passes([region("ZOO", confidence=0.90)]))

    def test_sfx_kecil_tidak_terpengaruh_ukuran_besar(self):
        # aturan glyph tidak pernah berlaku di bawah 100x100: perilaku lama
        self.assertTrue(passes([region("AA", width=60, height=40)]))


class GlyphRuleScopeTest(unittest.TestCase):
    """Di luar jangkauan aturan glyph, perilaku tidak berubah."""

    def test_dialog_panjang_di_kotak_besar_diterima(self):
        # 'FEAR DEATH.' = 9 karakter non-spasi -> aturan glyph tak berlaku
        self.assertTrue(passes([region("FEAR DEATH.", confidence=1.0,
                                       width=213, height=40)]))

    def test_kata_bertanda_baca_ganda_diterima(self):
        # jalur short_emotional_dialogue yang sudah ada: butuh >=2 tanda baca
        self.assertTrue(passes([region("YES!!", confidence=0.90,
                                       width=120, height=100)]))

    def test_kata_bertanda_baca_tunggal_kini_diterima_berkat_keyakinan(self):
        # "YES!" (1 tanda baca) tidak cocok pola short_emotional_dialogue, tapi
        # conf 0.90 >= 0.85 sehingga kini lolos. Dulu ia ditolak sebagai glyph.
        self.assertTrue(passes([region("YES!", confidence=0.90,
                                       width=120, height=100)]))

    def test_kata_bertanda_baca_tunggal_keyakinan_rendah_masih_ditolak(self):
        self.assertFalse(passes([region("YES!", confidence=0.80,
                                        width=120, height=100)]))

    def test_ambang_keyakinan_persis_085(self):
        # pakai kata non-narasi supaya menguji lever confidence murni:
        # 'AND' lolos lewat jalur narration_word yang memang tak peduli angka ini
        self.assertTrue(passes([region("ZOO", confidence=0.85)]))
        self.assertFalse(passes([region("ZOO", confidence=0.84)]))

    def test_kata_narasi_tidak_peduli_nilai_keyakinan(self):
        # selama masih di atas lantai trustworthy_region (0.75)
        self.assertTrue(passes([region("AND", confidence=0.84)]))
        self.assertFalse(passes([region("AND", confidence=0.70)]))

    def test_kata_sambung_multi_kata_di_luar_ukuran_kotak(self):
        # 'AND SO' -> 'ANDSO' = 5 karakter, ujian glyph tak berlaku
        self.assertTrue(passes([region("AND SO", width=200, height=110)]))


if __name__ == "__main__":
    unittest.main()
