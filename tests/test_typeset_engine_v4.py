import unittest
from PIL import Image, ImageDraw, ImageFont

from typeset_engine_v4 import (
    IndonesianHyphenator,
    compute_ellipse_line_budgets,
    evaluate_partition_quality,
    fit_text_v4,
    wrap_words_to_budgets,
)


class TypesetEngineV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.font_path = "assets/fonts/wildwords/CC Wild Words Roman.ttf"
        cls.font = ImageFont.truetype(cls.font_path, size=32)

    def test_indonesian_syllabification(self):
        syllables = IndonesianHyphenator.hyphenate_word("selamat")
        self.assertEqual(syllables, ["se", "la", "mat"])

        syllables2 = IndonesianHyphenator.hyphenate_word("dentingan")
        self.assertEqual(syllables2, ["den", "ting", "an"])

        syllables3 = IndonesianHyphenator.hyphenate_word("mendengarkannya")
        self.assertTrue(len(syllables3) >= 3)
        self.assertIn("men", syllables3[0].lower())

        candidates = IndonesianHyphenator.get_hyphenation_candidates("MENDENGARKANNYA")
        self.assertTrue(len(candidates) >= 1)
        self.assertTrue(candidates[0][0].endswith("-"))

    def test_ellipse_line_budgets_diamond_profile(self):
        budgets = compute_ellipse_line_budgets(
            bubble_w=300, bubble_h=300, line_count=3, line_height=32, spacing=5, margin_ratio=0.88
        )
        self.assertIsNotNone(budgets)
        self.assertEqual(len(budgets), 3)
        # Middle line must be strictly wider than top and bottom
        self.assertGreater(budgets[1], budgets[0])
        self.assertGreater(budgets[1], budgets[2])

    def test_anti_tower_wrapping(self):
        # 6 words: "ANDA HEBAT SEKALI, TUAN MUDA GLENN."
        # In a 3-line budget, it must avoid 1-word-per-line tower
        words = "ANDA HEBAT SEKALI, TUAN MUDA GLENN.".split()
        budgets = [240, 260, 240]
        lines = wrap_words_to_budgets(words, self.font, budgets)
        self.assertIsNotNone(lines)
        self.assertEqual(len(lines), 3)
        # In 3 lines with 6 words, each line should have ~2 words
        word_counts = [len(l.split()) for l in lines]
        self.assertEqual(word_counts, [2, 2, 2])

    def test_fit_text_v4_short_dialogue_avoids_vertical_tower(self):
        # In the old engine, this became 5 lines of 1 word each ("SELAMAT \n PAGI, \n TUAN \n MUDA \n GLENN.")
        # In V4, it must be 2 or 3 lines and maintain healthy font size >= 24
        img = Image.new("RGB", (400, 400), "white")
        draw = ImageDraw.Draw(img)

        result = fit_text_v4(
            draw=draw,
            text="SELAMAT PAGI, TUAN MUDA GLENN.",
            font_path=self.font_path,
            width=240,
            height=290,
            minimum=16,
            maximum=42,
            spacing_ratio=0.14,
            is_curved_balloon=True,
        )
        self.assertIsNotNone(result)
        self.assertLessEqual(result["line_count"], 3)
        self.assertGreaterEqual(result["font_size"], 24)
        wrapped_lines = result["wrapped"].split("\n")
        self.assertLessEqual(len(wrapped_lines), 3)

    def test_fit_text_v4_avoids_microscopic_squashing(self):
        # Text with long words: "AKU BISA MENDENGAR DENTINGAN PEDANG..."
        # In the old engine, font collapsed to 23px. In V4, font stays >= 28px.
        img = Image.new("RGB", (400, 400), "white")
        draw = ImageDraw.Draw(img)

        result = fit_text_v4(
            draw=draw,
            text="AKU BISA MENDENGAR DENTINGAN PEDANG...",
            font_path=self.font_path,
            width=230,
            height=300,
            minimum=16,
            maximum=42,
            spacing_ratio=0.14,
            is_curved_balloon=True,
        )
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result["font_size"], 28)
        self.assertLessEqual(result["line_count"], 4)


if __name__ == "__main__":
    unittest.main()
