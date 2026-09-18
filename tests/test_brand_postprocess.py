import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from brand_postprocess import apply_branding_legacy as apply_branding, credit_regions, replace_credit_cluster, trim_source_footer, trim_source_header


class BrandingTests(unittest.TestCase):
    def test_trims_complete_scanlator_card_from_first_page(self):
        image = Image.new("RGBA", (800, 12000), "white")
        regions = [
            {"text": "OI SCANS", "bbox": {"x": 30, "y": 20, "width": 120, "height": 30}},
            {"text": "CHAPTERS UP TO 20", "bbox": {"x": 400, "y": 70, "width": 250, "height": 45}},
            {"text": "QISCANS", "bbox": {"x": 100, "y": 100, "width": 180, "height": 55}},
            {"text": "FOLLOW FOR FASTER UPDATES", "bbox": {"x": 430, "y": 180, "width": 280, "height": 40}},
            {"text": "DISCORD.GG/SOURCE", "bbox": {"x": 540, "y": 760, "width": 220, "height": 35}},
        ]
        trimmed, boundary = trim_source_header(image, regions)
        self.assertIsNotNone(boundary)
        self.assertEqual(trimmed.height, image.height - boundary)

    def test_trims_large_scanlator_card_above_bottom_threshold(self):
        image = Image.new("RGBA", (690, 1150), "white")
        image.paste((8, 4, 16, 255), (0, 441, 690, 1150))
        regions = [{
            "text": "THUNDER SCANS",
            "bbox": {"x": 80, "y": 510, "width": 554, "height": 349},
        }]
        trimmed, boundary = trim_source_footer(image, regions)
        self.assertEqual(boundary, 441)
        self.assertEqual(trimmed.size, (690, 441))

    def test_trims_promo_card_when_ocr_splits_credit_into_lines(self):
        image = Image.new("RGBA", (690, 1150), "white")
        regions = [
            {"text": "THUNDER SCANS", "bbox": {"x": 90, "y": 510, "width": 500, "height": 120}},
            {"text": "DISCORD.GG/SOURCE", "bbox": {"x": 300, "y": 970, "width": 300, "height": 35}},
            {"text": "EN-THUNDERSCANS.COM", "bbox": {"x": 285, "y": 1060, "width": 330, "height": 35}},
        ]
        trimmed, boundary = trim_source_footer(image, regions)
        self.assertEqual(boundary, 441)
        self.assertEqual(trimmed.size, (690, 441))

    def test_thunderscans_credit_variants_are_detected_as_one_promo(self):
        regions = [
            {"text": "READ AT EN-THUNDERSCANS.COM", "bbox": {"x": 20, "y": 40, "width": 280, "height": 40}},
            {"text": "YOU CAN READ THE CHAPTER ON", "bbox": {"x": 30, "y": 120, "width": 220, "height": 20}},
            {"text": "EN-THUNDERSCANS.COM", "bbox": {"x": 30, "y": 145, "width": 180, "height": 20}},
        ]
        self.assertEqual(credit_regions({"regions": regions}), regions)

    def test_credit_logo_is_skipped_when_it_would_cover_dialogue(self):
        image = Image.new("RGBA", (500, 600), "white")
        logo = Image.new("RGBA", (300, 120), (100, 20, 220, 255))
        credit = [{"text": "EVASCANS.ORG", "bbox": {"x": 20, "y": 220, "width": 140, "height": 35}}]
        dialogue = [{"text": "REAL DIALOGUE", "bbox": {"x": 150, "y": 205, "width": 220, "height": 100}}]
        before = image.copy()
        self.assertIsNone(replace_credit_cluster(image, credit, logo, dialogue))
        self.assertEqual(list(image.getdata()), list(before.getdata()))

    def test_clears_top_card_replaces_credit_and_trims_footer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_path = root / "page.png"
            ocr_path = root / "ocr.json"
            logo_path = root / "logo.png"
            text_path = root / "text.png"
            banner_path = root / "banner.png"
            page = Image.new("RGB", (400, 1200), "white")
            page.paste((20, 10, 30), (0, 0, 400, 220))
            page.paste((90, 10, 20), (0, 980, 400, 1200))
            page.save(image_path)
            Image.new("RGBA", (160, 70), (100, 20, 220, 255)).save(logo_path)
            Image.new("RGBA", (240, 60), (100, 20, 220, 100)).save(text_path)
            Image.new("RGBA", (400, 160), (20, 10, 40, 255)).save(banner_path)
            regions = [
                {"text": "EVASCANS", "bbox": {"x": 80, "y": 30, "width": 200, "height": 40}},
                {"text": "discord.gg/source", "bbox": {"x": 90, "y": 100, "width": 180, "height": 30}},
                {"text": "READ THIS SERIES ONLY AT", "bbox": {"x": 20, "y": 500, "width": 180, "height": 20}},
                {"text": "EVASCANS.ORG", "bbox": {"x": 30, "y": 530, "width": 140, "height": 20}},
                {"text": "SCANS", "bbox": {"x": 50, "y": 1100, "width": 120, "height": 40}},
            ]
            ocr_path.write_text(json.dumps({"regions": regions}), encoding="utf-8")
            (root / "translations.json").write_text(json.dumps({"groups": [{
                "translation": "DIALOGUE", "bbox": {"x": 150, "y": 500, "width": 220, "height": 80},
            }]}), encoding="utf-8")

            report = apply_branding(
                image_path, ocr_path, logo_path, text_path,
                first_page=True, banner_path=banner_path,
            )

            self.assertIsNotNone(report["top_credit_cleared"])
            self.assertIsInstance(report["replacements"], list)
            self.assertIsNotNone(report["source_footer_trimmed_at"])
            with Image.open(image_path) as result:
                expected = report["source_footer_trimmed_at"] + 160
                self.assertEqual(result.size, (400, expected))

    def test_translation_group_bbox_drives_large_credit_trim(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image_path, ocr_path = root / "page.png", root / "ocr.json"
            logo_path, text_path, banner_path = root / "logo.png", root / "text.png", root / "banner.png"
            Image.new("RGB", (690, 1150), "white").save(image_path)
            ocr_path.write_text(json.dumps({"regions": []}), encoding="utf-8")
            (root / "translations.json").write_text(json.dumps({"groups": [{
                "text": "THUNDER SCANS", "translation": "THUNDER SCANS",
                "bbox": {"x": 80, "y": 510, "width": 554, "height": 349},
            }]}), encoding="utf-8")
            Image.new("RGBA", (300, 100), (100, 20, 220, 255)).save(logo_path)
            Image.new("RGBA", (240, 60), (100, 20, 220, 100)).save(text_path)
            Image.new("RGBA", (690, 160), (20, 10, 40, 255)).save(banner_path)
            report = apply_branding(image_path, ocr_path, logo_path, text_path, banner_path=banner_path)
            self.assertEqual(report["source_footer_trimmed_at"], 441)
            with Image.open(image_path) as result:
                self.assertEqual(result.height, 601)


if __name__ == "__main__":
    unittest.main()
