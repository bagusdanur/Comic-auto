import unittest

from typeset_test import crosses_stacked_card_boundary, prefer_v2_over_contour, safe_surface_bbox


class TypesetGeometryTests(unittest.TestCase):
    def test_round_balloon_keeps_contour_instead_of_full_white_panel(self):
        contour = {"x": 181, "y": 4700, "width": 312, "height": 227}
        self.assertFalse(prefer_v2_over_contour(contour, True, True))

    def test_implausible_tall_edge_contour_uses_v2(self):
        contour = {"x": 226, "y": 120, "width": 222, "height": 528}
        self.assertTrue(prefer_v2_over_contour(contour, True, False))

    def test_rejects_tall_expansion_from_wide_caption_card(self):
        raw = {"x": 316, "y": 5334, "width": 376, "height": 135}
        merged_cards = {"x": 198, "y": 5055, "width": 494, "height": 433}
        self.assertTrue(crosses_stacked_card_boundary(raw, merged_cards, True))

    def test_keeps_normal_bubble_expansion(self):
        raw = {"x": 120, "y": 180, "width": 190, "height": 105}
        bubble = {"x": 90, "y": 140, "width": 250, "height": 190}
        self.assertFalse(crosses_stacked_card_boundary(raw, bubble))

    def test_keeps_wide_dialogue_without_straight_card_border(self):
        raw = {"x": 100, "y": 200, "width": 320, "height": 100}
        merged = {"x": 20, "y": 20, "width": 600, "height": 500}
        self.assertFalse(crosses_stacked_card_boundary(raw, merged, False))

    def test_keeps_wide_card_with_modest_gutter(self):
        raw = {"x": 100, "y": 200, "width": 320, "height": 100}
        card = {"x": 80, "y": 180, "width": 360, "height": 145}
        self.assertFalse(crosses_stacked_card_boundary(raw, card))

    def test_uses_high_confidence_v2_bubble_surface(self):
        raw = {"x": 200, "y": 4608, "width": 270, "height": 79}
        surface = {
            "surface_type": "BUBBLE_FLAT", "confidence": 0.938,
            "collides_with": [],
            "safe_polygon": [[648, 4542], [648, 4753], [41, 4753], [41, 4542]],
        }
        self.assertEqual(
            safe_surface_bbox(surface, raw),
            {"x": 41, "y": 4542, "width": 607, "height": 211},
        )

    def test_rejects_ambiguous_v2_surface(self):
        raw = {"x": 10, "y": 10, "width": 100, "height": 40}
        surface = {
            "surface_type": "BUBBLE_FLAT", "confidence": 0.7,
            "collides_with": [], "safe_polygon": [[0, 0], [300, 0], [300, 200], [0, 200]],
        }
        self.assertIsNone(safe_surface_bbox(surface, raw))


if __name__ == "__main__":
    unittest.main()
