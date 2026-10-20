import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from quality_pipeline_v2 import analyze, validate


class QualityPipelineV2Test(unittest.TestCase):
    def test_confirmed_automatic_promotion_cleanup_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.full((240, 320, 3), 180, dtype=np.uint8)
            cleaned = original.copy()
            rendered = original.copy()
            rendered[80:145, 0:190] = 235
            for name, value in (("original.png", original), ("cleaned.png", cleaned), ("rendered.png", rendered)):
                Image.fromarray(value).save(root / name)
            Image.fromarray(np.zeros((240, 320), dtype=np.uint8)).save(root / "mask.png")
            (root / "surface.json").write_text(json.dumps({"surfaces": [], "ambiguous_count": 0}))
            (root / "typeset-report.json").write_text(json.dumps({
                "results": [],
                "automatic_promotions": [{
                    "action": "REMOVED_PROMOTION",
                    "bbox": {"x": 14, "y": 92, "width": 160, "height": 39},
                    "affected_bbox": {"x": 0, "y": 80, "width": 190, "height": 65},
                }],
            }))
            report = validate(
                root / "original.png", root / "cleaned.png", root / "rendered.png",
                root / "mask.png", root / "surface.json", root / "quality.json",
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["checks"]["outside_allowed_pixels"], 0)

    def test_inpainting_context_around_long_promo_mask_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.full((240, 320, 3), 180, dtype=np.uint8)
            cleaned = original.copy()
            rendered = original.copy()
            cleaned[85:126, 50:271] = 170
            rendered[85:126, 50:271] = 170
            mask = np.zeros((240, 320), dtype=np.uint8)
            mask[100:111, 70:251] = 255
            for name, value in (("original.png", original), ("cleaned.png", cleaned), ("rendered.png", rendered)):
                Image.fromarray(value).save(root / name)
            Image.fromarray(mask).save(root / "mask.png")
            (root / "surface.json").write_text(json.dumps({"surfaces": [], "ambiguous_count": 0}))
            report = validate(
                root / "original.png", root / "cleaned.png", root / "rendered.png",
                root / "mask.png", root / "surface.json", root / "quality.json",
            )
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["checks"]["outside_allowed_pixels"], 0)

    def test_surface_analysis_and_clean_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (320, 240), (30, 35, 45))
            draw = ImageDraw.Draw(image)
            draw.rectangle((65, 70, 255, 165), fill=(250, 250, 250))
            original = root / "original.png"
            cleaned = root / "cleaned.png"
            rendered = root / "rendered.png"
            image.save(original)
            image.save(cleaned)
            image.save(rendered)
            ocr = {
                "regions": [{
                    "id": "r1", "bbox": {"x": 105, "y": 100, "width": 110, "height": 28},
                    "polygon": [[105, 100], [215, 100], [215, 128], [105, 128]],
                }]
            }
            translations = {"groups": [{
                "id": "g1", "bbox": {"x": 105, "y": 100, "width": 110, "height": 28},
                "source_region_ids": ["r1"], "translation": "CONTOH", "text": "SAMPLE",
            }]}
            ocr_path, translations_path = root / "ocr.json", root / "translations.json"
            ocr_path.write_text(json.dumps(ocr), encoding="utf-8")
            translations_path.write_text(json.dumps(translations), encoding="utf-8")
            surface_path = root / "surface.json"
            surface = analyze(original, ocr_path, translations_path, surface_path)
            self.assertEqual(surface["surface_count"], 1)
            self.assertTrue(surface["surfaces"][0]["expanded_from_raw"])
            mask = root / "mask.png"
            Image.fromarray(np.zeros((240, 320), dtype=np.uint8)).save(mask)
            report = validate(original, cleaned, rendered, mask, surface_path, root / "quality.json")
            self.assertEqual(report["status"], "PASS")
            self.assertTrue(report["checks"]["dimensions_ok"])

    def _dark_bubble_fixture(self, root: Path, *, output_changes_text: bool,
                             mask_covers_text: bool, bright_text: bool) -> Path:
        """Gelap (bg 24) berisi teks + garis artwork putih di dalam poligon."""
        original = np.full((240, 320, 3), 24, dtype=np.uint8)
        original[60:110, 60:160] = 0 if not bright_text else 200      # teks
        original[140:190, 200:260] = 220                              # artwork
        cleaned = original.copy()
        cleaned[60:110, 60:160] = 24                                  # REMOVAL
        rendered = cleaned.copy()
        if output_changes_text:
            rendered[60:110, 60:160] = 120                            # terjemahan
        else:
            rendered[60:110, 60:160] = original[60:110, 60:160]       # sisa teks
        mask = np.zeros((240, 320), dtype=np.uint8)
        if mask_covers_text:
            mask[60:110, 60:160] = 255
        for name, value in (("original.png", original), ("cleaned.png", cleaned), ("rendered.png", rendered)):
            Image.fromarray(value).save(root / name)
        Image.fromarray(mask).save(root / "mask.png")
        (root / "surface.json").write_text(json.dumps({
            "ambiguous_count": 0,
            "surfaces": [{
                "id": "bubble-dark",
                "surface_type": "CHAT_UI",
                "polygon": [[40, 40], [280, 40], [280, 200], [40, 200]],
                "raw_polygon": [[40, 40], [280, 40], [280, 200], [40, 200]],
                "stats": {"median_rgb": [24, 24, 24]},
            }],
        }))
        (root / "typeset-report.json").write_text(json.dumps({"results": [], "automatic_promotions": []}))
        return root / "quality.json"

    def test_dark_bubble_untouched_artwork_is_not_raw_residual(self):
        # Regresi bug nyata: teks hitam di bubble gelap jatuh di bawah ambang
        # |abu-24|>=45, sehingga garis artwork putih yang tak tersentuh yang
        # terukur dan dianggap sisa teks. Terjemahan sudah bersih tetap FAIL.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = self._dark_bubble_fixture(
                root, output_changes_text=True, mask_covers_text=True, bright_text=False,
            )
            report = validate(root / "original.png", root / "cleaned.png", root / "rendered.png",
                              root / "mask.png", root / "surface.json", report_path)
            self.assertEqual(report["checks"]["residual_candidates"], [])
            self.assertNotIn("RAW_TEXT_RESIDUAL", report["review_reasons"])
            self.assertEqual(report["status"], "PASS")

    def test_real_raw_text_residual_still_fails(self):
        # Gerbang tetap keras: kalau keluaran masih memperlihatkan piksel teks
        # asli apa adanya, QC wajib tetap menolak.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = self._dark_bubble_fixture(
                root, output_changes_text=False, mask_covers_text=True, bright_text=False,
            )
            report = validate(root / "original.png", root / "cleaned.png", root / "rendered.png",
                              root / "mask.png", root / "surface.json", report_path)
            self.assertIn("RAW_TEXT_RESIDUAL", report["review_reasons"])
            self.assertEqual(report["status"], "FAIL")

    def test_fallback_catches_residual_when_removal_mask_misses_surface(self):
        # Bila mask removal tidak menutup permukaan ini, jangan dilewati: uji
        # cadangan berbasis kecerahan wajib masih menangkap sisa teks.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = self._dark_bubble_fixture(
                root, output_changes_text=False, mask_covers_text=False, bright_text=True,
            )
            report = validate(root / "original.png", root / "cleaned.png", root / "rendered.png",
                              root / "mask.png", root / "surface.json", report_path)
            self.assertIn("RAW_TEXT_RESIDUAL", report["review_reasons"])
            self.assertEqual(report["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
