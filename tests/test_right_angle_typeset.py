"""Test untuk caption 90 derajat dan guard typeset.

Fitur `894948e` (caption sudut penuh) dan perbaikan `fe52ec7` (guard yang
ikut di-rebase) tidak punya test sama sekali sampai regresi `page_safe_mask`
menumbangkan satu chapter utuh -- test lama hanya menguji fungsi bantu dan
tidak pernah menjalankan jalur ini. Setiap test di sini menutup satu kasus
yang pernah gagal.
"""
import math
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from typeset_engine_v5 import (
    fit_text_v5,
    paint_fitted_text,
    right_angle_fit_space,
)
from typeset_test import source_rotation


FONT = "assets/fonts/wildwords/CC Wild Words Roman.ttf"
TEXT = "TUNGGU SEBENTAR LAGI."


def tall_region(orientation_angle=1, width=140, height=900):
    return {"bbox": {"x": 100, "y": 2000, "width": width, "height": height},
            "polygon": [[100, 2000], [100 + width, 2000], [100 + width, 2000 + height], [100, 2000 + height]],
            "orientation_angle": orientation_angle,
            "text": "JUST WAIT A LITTLE LONGER."}


class SourceRotationTests(unittest.TestCase):
    """source_rotation harus mengenali sudut penuh, bukan hanya kemiringan."""

    def test_vertical_caption_paints_at_minus_90(self):
        # PaddleOCR men-deskew kotak tinggi dengan np.rot90 lalu angle head
        # memutuskan flip 180; angle==1 berarti pembaca memutar halaman
        # searah jarum jam, jadi teks harus digambar di -90.
        self.assertEqual(source_rotation([tall_region(orientation_angle=1)]), -90.0)

    def test_vertical_caption_with_opposite_reading_paints_at_plus_90(self):
        self.assertEqual(source_rotation([tall_region(orientation_angle=0)]), 90.0)

    def test_vertical_caption_without_angle_data_falls_back(self):
        # Laporan OCR lama tidak punya orientation_angle.
        region = tall_region()
        del region["orientation_angle"]
        self.assertEqual(source_rotation([region]), -90.0)

    def test_horizontal_and_jitter_stay_unrotated(self):
        flat = {"bbox": {"x": 0, "y": 0, "width": 300, "height": 40},
                "polygon": [[0, 0], [300, 3], [300, 43], [0, 40]]}
        self.assertEqual(source_rotation([flat]), 0.0)

    def test_tilted_card_keeps_its_measured_slant(self):
        # Sisis atas membentuk atan2(63, 300) = 11.86 derajat dan harus
        # dibawa utuh, bukan dibulatkan dan bukan dilepas ke 0.
        tilted = {"bbox": {"x": 0, "y": 0, "width": 300, "height": 40},
                  "polygon": [[0, 0], [300, 63], [300, 103], [0, 40]]}
        self.assertAlmostEqual(source_rotation([tilted]), math.degrees(math.atan2(63, 300)), places=2)

    def test_majority_rules_when_a_group_mixes_shapes(self):
        mixed = [tall_region(orientation_angle=1), tall_region(orientation_angle=1),
                 {"bbox": {"x": 0, "y": 0, "width": 300, "height": 40},
                  "polygon": [[0, 0], [300, 3], [300, 43], [0, 40]]}]
        self.assertEqual(source_rotation(mixed), -90.0)


class FitSpaceTests(unittest.TestCase):
    """right_angle_fit_space adalah invers dari putaran paint_fitted_text."""

    def test_non_right_angle_returns_inputs_untouched(self):
        mask = np.full((50, 30), 255, np.uint8)
        anchor = (10.0, 20.0)
        for rotation in (0.0, 12.0, -24.0):
            out_mask, out_anchor = right_angle_fit_space(mask, anchor, (0, 0, 30, 50), rotation)
            self.assertIs(out_mask, mask)
            self.assertEqual(out_anchor, anchor)

    def test_right_angle_transposes_the_mask(self):
        mask = np.full((900, 140), 255, np.uint8)
        out, anchor = right_angle_fit_space(mask, (70.0, 450.0), (0, 0, 140, 900), -90.0)
        self.assertEqual(out.shape, (140, 900))
        self.assertEqual(anchor, (899 - 450, 70.0))

    def test_footprint_clips_the_fit_to_the_raw_box(self):
        # Surface 1576 px tinggi untuk caption 1094 px: tanpa clipping teks
        # menjalar ke artwork dan kena guard QC.
        surface = np.full((1576, 154), 255, np.uint8)
        footprint = {"x": 0, "y": 241, "width": 154, "height": 1094}
        out, _ = right_angle_fit_space(surface, (77.0, 788.0), (0, 0, 154, 1576),
                                       -90.0, footprint=footprint)
        # np.rot90 menukar dimensi: kanvas jadi (page_w, page_h) sebagai
        # bentuk numpy, yaitu PIL size (page_h, page_w) -> muat teks horizontal.
        self.assertEqual(out.shape, (154, 1576))
        # area yang diizinkan tepat = footprint 1094 baris, dan itu sudah
        # jauh lebih kecil daripada surface 1576 baris yang membawanya.
        allowed = int(np.count_nonzero(out))
        self.assertEqual(allowed, 1094 * 154)
        self.assertLess(allowed, 1576 * 154)


class RightAnglePaintTests(unittest.TestCase):
    """Teks 90 derajat harus mendarat di dalam footprint mentah."""

    def paint(self, rotation):
        page_h, page_w = 900, 140
        page_mask = np.full((page_h, page_w), 255, np.uint8)
        x1, y1 = 100, 2000
        bounds = (x1, y1, x1 + page_w, y1 + page_h)
        anchor_center = (page_w / 2, page_h / 2)
        fit_mask, fit_anchor = right_angle_fit_space(page_mask, anchor_center, bounds, rotation)

        fitted = fit_text_v5(TEXT, FONT, fit_mask, 22, 80, 2, anchor_center=fit_anchor, alignment="center")
        self.assertIsNotNone(fitted, "caption 90 derajat gagal di-fit di footprint-nya")

        page = Image.new("RGB", (1200, 3000), (240, 240, 240))
        painted = paint_fitted_text(
            page, bounds, fitted, {"color": (0, 0, 0), "fill_mode": "solid"},
            2, (255, 255, 255), rotation=rotation, rotation_center=fit_anchor,
            safe_mask=page_mask,
        )
        arr = np.asarray(page)
        region = arr[y1:y1 + page_h, x1:x1 + page_w]
        dark = region.max(axis=2) < 120
        return painted, int(np.count_nonzero(dark & (page_mask == 0))), int(np.count_nonzero(dark))

    def test_minus_90_keeps_every_glyph_inside_the_footprint(self):
        painted, outside, total = self.paint(-90.0)
        self.assertTrue(painted)
        self.assertGreater(total, 0)
        self.assertEqual(outside, 0)

    def test_plus_90_keeps_every_glyph_inside_the_footprint(self):
        painted, outside, total = self.paint(90.0)
        self.assertTrue(painted)
        self.assertGreater(total, 0)
        self.assertEqual(outside, 0)

    def test_glyphs_outside_the_guard_are_refused(self):
        # Guard yang lebih sempit daripada kanvas => paint harus menolak,
        # bukan mengecat di luar surface.
        page_h, page_w = 900, 140
        page_mask = np.full((page_h, page_w), 255, np.uint8)
        bounds = (100, 2000, 100 + page_w, 2000 + page_h)
        fit_mask, fit_anchor = right_angle_fit_space(page_mask, (70.0, 450.0), bounds, -90.0)
        fitted = fit_text_v5(TEXT, FONT, fit_mask, 22, 80, 2, anchor_center=fit_anchor, alignment="center")
        self.assertIsNotNone(fitted)
        tiny = np.zeros_like(page_mask)
        page = Image.new("RGB", (1200, 3000), (240, 240, 240))
        self.assertFalse(paint_fitted_text(
            page, bounds, fitted, {"color": (0, 0, 0), "fill_mode": "solid"},
            2, (255, 255, 255), rotation=-90.0, rotation_center=fit_anchor,
            safe_mask=tiny,
        ))


class GuardShapeRegressionTests(unittest.TestCase):
    """Regresi fe52ec7: guard harus ikut bounds, dan kalau tidak -> error jelas.

    Sebelum perbaikan ini ketidakcocokan bentuk muncul sebagai
    `ValueError: operands could not be broadcast together with shapes
    (96,527) (268,690)` di tengah chapter dan menjatuhkan seluruh job.
    """

    def mismatched_guard(self):
        page_mask = np.full((900, 140), 255, np.uint8)
        bounds = (100, 2000, 100 + 140, 2000 + 900)
        fit_mask, fit_anchor = right_angle_fit_space(page_mask, (70.0, 450.0), bounds, -90.0)
        fitted = fit_text_v5(TEXT, FONT, fit_mask, 22, 80, 2, anchor_center=fit_anchor, alignment="center")
        return page_mask, bounds, fitted, fit_anchor

    def test_stale_guard_raises_a_named_invariant(self):
        page_mask, bounds, fitted, fit_anchor = self.mismatched_guard()
        self.assertIsNotNone(fitted)
        stale = np.full((268, 690), 255, np.uint8)   # surface lama sebelum bounds di-rebase
        page = Image.new("RGB", (1200, 3000), (240, 240, 240))
        with self.assertRaises(RuntimeError) as caught:
            paint_fitted_text(
                page, bounds, fitted, {"color": (0, 0, 0), "fill_mode": "solid"},
                2, (255, 255, 255), rotation=-90.0, rotation_center=fit_anchor,
                safe_mask=stale,
            )
        message = str(caught.exception)
        self.assertIn("does not match bounds", message)
        self.assertIn("re-point the page-space surface", message)

    def test_matching_guard_never_raises(self):
        page_mask, bounds, fitted, fit_anchor = self.mismatched_guard()
        self.assertIsNotNone(fitted)
        page = Image.new("RGB", (1200, 3000), (240, 240, 240))
        try:
            paint_fitted_text(
                page, bounds, fitted, {"color": (0, 0, 0), "fill_mode": "solid"},
                2, (255, 255, 255), rotation=-90.0, rotation_center=fit_anchor,
                safe_mask=page_mask,
            )
        except RuntimeError as exc:
            self.fail(f"guard yang benar seharusnya tidak error: {exc}")


if __name__ == "__main__":
    unittest.main()
