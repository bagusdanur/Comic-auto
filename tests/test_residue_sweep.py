import unittest

import cv2
import numpy as np

from remove_text_test import sweep_residue_fragments


class ResidueSweepTests(unittest.TestCase):
    """Pecahan tinta kecil di dalam bbox grup yang sudah dibersihkan harus hilang.

    pyscho ch2 page-008 bubble-001: apostrof dan batang sempit lolos dari mask
    white-fill lalu tertinggal sebagai garis hitam di balon putih.
    """

    def build(self, status="CLEANED"):
        original = np.full((160, 220, 3), 255, np.uint8)
        cv2.line(original, (100, 40), (104, 52), (0, 0, 0), 3)   # apostrof sisa
        cv2.rectangle(original, (60, 80), (72, 110), (0, 0, 0), -1)  # batang sisa
        output = original.copy()
        output[80:110, 60:72] = 255                               # batang sudah bersih
        results = [{"id": "bubble-001", "status": status}]
        boxes = {"bubble-001": {"x": 50, "y": 30, "width": 90, "height": 90}}
        allowed = np.zeros(original.shape[:2], np.uint8)
        out, allowed, cleaned = sweep_residue_fragments(original, output, results, boxes, allowed)
        return original, output, out, allowed, cleaned, results

    def test_sisa_apostrof_hilang(self):
        original, initial, out, allowed, cleaned, results = self.build()
        self.assertGreater(cleaned, 0)
        self.assertGreater(int(out[44, 102].min()), 200)
        self.assertEqual(int(original[44, 102].min()), 0)
        self.assertGreater(int(results[0].get("residue_pixels", 0)), 0)
        changed = np.any(out != initial, axis=2)
        self.assertFalse(np.any(changed & (allowed == 0)))

    def test_grup_preserved_tidak_disentuh(self):
        original, initial, out, allowed, cleaned, results = self.build(status="PRESERVED")
        self.assertEqual(cleaned, 0)
        self.assertTrue(np.array_equal(out, initial))
        self.assertFalse(allowed.any())

    def test_busur_panjang_tidak_disapu(self):
        original = np.full((160, 220, 3), 255, np.uint8)
        cv2.line(original, (20, 100), (200, 100), (0, 0, 0), 3)
        output = original.copy()
        results = [{"id": "bubble-002", "status": "INPAINTED"}]
        boxes = {"bubble-002": {"x": 60, "y": 60, "width": 100, "height": 80}}
        allowed = np.zeros(original.shape[:2], np.uint8)
        out, _, cleaned = sweep_residue_fragments(original, output, results, boxes, allowed)
        self.assertEqual(cleaned, 0)
        self.assertEqual(int(out[100, 100].min()), 0)

    def test_residue_berwarna_biru_disapu(self):
        """Ghosting teks sumber BERWARNA (min channel >= 100) harus ikut tersapu.

        Bug horror ch15 page-007 bubble-016: teks biru tertinggal di belakang
        terjemahan karena gerbang lama hanya menangkap tinta gelap (min < 100).
        """
        original = np.full((160, 220, 3), 255, np.uint8)
        cv2.rectangle(original, (60, 60), (74, 95), (110, 120, 207), -1)  # biru, min=110
        output = original.copy()  # inpaint tidak menyentuhnya -> unchanged
        results = [{"id": "bubble-003", "status": "INPAINTED"}]
        boxes = {"bubble-003": {"x": 50, "y": 40, "width": 90, "height": 90}}
        allowed = np.zeros(original.shape[:2], np.uint8)
        out, allowed, cleaned = sweep_residue_fragments(original, output, results, boxes, allowed)
        self.assertGreater(cleaned, 0)
        self.assertGreater(int(out[75, 67].min()), 200)
        self.assertGreater(int(results[0].get("residue_pixels", 0)), 0)
        changed = np.any(out != output, axis=2)
        self.assertFalse(np.any(changed & (allowed == 0)))

    def test_artwork_berwarna_besar_tidak_disapu(self):
        """Region artwork berwarna yang besar tetap dilindungi guard komponen."""
        original = np.full((200, 260, 3), 255, np.uint8)
        cv2.rectangle(original, (60, 50), (160, 150), (200, 50, 50), -1)  # area 10000
        output = original.copy()
        results = [{"id": "bubble-004", "status": "INPAINTED"}]
        boxes = {"bubble-004": {"x": 50, "y": 40, "width": 130, "height": 130}}
        allowed = np.zeros(original.shape[:2], np.uint8)
        out, _, cleaned = sweep_residue_fragments(original, output, results, boxes, allowed)
        self.assertEqual(cleaned, 0)
        self.assertEqual(int(out[100, 100].min()), 50)
