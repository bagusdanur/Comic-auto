import unittest

from translate import is_promo_only_page


class PromoPassthroughTests(unittest.TestCase):
    def test_source_warning_above_artwork_is_safe_promo_only_page(self):
        ocr = {
            "height": 6000,
            "regions": [
                {"text": "WARNING", "bbox": {"x": 20, "y": 80, "width": 200, "height": 50}},
                {"text": "PLEASE READ THIS SERIES ON OUR WEBSITE EN-THUNDERSCANS.COM",
                 "bbox": {"x": 20, "y": 300, "width": 500, "height": 80}},
            ],
        }
        self.assertTrue(is_promo_only_page(ocr))

    def test_dialogue_later_on_page_is_not_promo_only(self):
        ocr = {
            "height": 6000,
            "regions": [
                {"text": "READ AT SOURCE.COM", "bbox": {"x": 20, "y": 80, "width": 200, "height": 40}},
                {"text": "WHAT HAPPENED?", "bbox": {"x": 200, "y": 3200, "width": 250, "height": 80}},
            ],
        }
        self.assertFalse(is_promo_only_page(ocr))


if __name__ == "__main__":
    unittest.main()
