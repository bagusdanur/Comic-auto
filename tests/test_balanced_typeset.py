import unittest

from PIL import ImageFont

from typeset_test import balanced_wrap_text, wrap_text


class BalancedTypesetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font = ImageFont.truetype(
            "assets/fonts/wildwords/CC Wild Words Roman.ttf", size=28
        )

    def test_balancing_preserves_line_count_and_reduces_ragged_width(self):
        text = "AKU BENAR BENAR TIDAK TAHU APA YANG HARUS KULAKUKAN"
        greedy = wrap_text(text, self.font, 260)
        balanced = balanced_wrap_text(text, self.font, 260)
        self.assertEqual(len(balanced), len(greedy))
        greedy_widths = [self.font.getlength(line) for line in greedy]
        balanced_widths = [self.font.getlength(line) for line in balanced]
        self.assertLessEqual(max(balanced_widths) - min(balanced_widths),
                             max(greedy_widths) - min(greedy_widths))

    def test_single_line_is_unchanged(self):
        self.assertEqual(
            balanced_wrap_text("TUNGGU!", self.font, 300),
            wrap_text("TUNGGU!", self.font, 300),
        )


if __name__ == "__main__":
    unittest.main()
