import unittest
from types import SimpleNamespace
import numpy as np
from source_typography import source_typography,select_font,italic_run_reconciliation


class SourceTypographyTests(unittest.TestCase):
    def setUp(self):
        self.args=SimpleNamespace(font='cc-roman',font_dialogue='cc-roman',font_narration='cc-roman',
            font_thought='cc-italic',font_shout='cc-bold-italic',font_sfx='cc-bold-italic',font_chat='cc-roman')

    def test_semantic_styles_render_with_one_regular_face(self):
        image=np.zeros((20,40,3),np.uint8);box={'x':0,'y':0,'width':40,'height':20}
        thought=source_typography(image,image,box,'thought')
        shout=source_typography(image,image,box,'shout')
        self.assertEqual(select_font(self.args,thought),'cc-roman')
        self.assertEqual(select_font(self.args,shout),'cc-roman')

    def test_dialogue_defaults_to_roman_when_visual_signal_is_empty(self):
        image=np.zeros((20,40,3),np.uint8);box={'x':0,'y':0,'width':40,'height':20}
        style=source_typography(image,image,box,'dialogue')
        self.assertEqual(style['font_variant'],'regular')
        self.assertEqual(select_font(self.args,style),'cc-roman')

    def test_raw_regular_overrides_semantic_shout_font(self):
        style={'semantic_style':'shout','font_variant':'regular'}
        self.assertEqual(select_font(self.args,style),'cc-roman')

    def test_bold_raw_signal_does_not_change_render_font(self):
        style={'semantic_style':'shout','font_variant':'bold'}
        self.assertEqual(select_font(self.args,style),'cc-roman')
    @staticmethod
    def _run(specs):
        """specs: (y, font_variant, raw_italic_score, semantic_style)"""
        return [{"id": f"g{y}", "bbox": {"x": 0, "y": y, "width": 60, "height": 40},
                 "surface_type": "BUBBLE_FLAT", "semantic_style": semantic,
                 "font_variant": variant, "raw_italic_score": score}
                for y, variant, score, semantic in specs]

    def test_gated_slant_joins_its_italic_neighbour(self):
        # "AND I'VE" terukur 0.667 tapi ditolak gerbang 4 garis Hough; saudara
        # sekalimat yang lolos sudah italic -> keduanya harus satu font.
        items=self._run([(0,"regular",0.667,"dialogue"),(52,"italic",0.455,"dialogue")])
        promoted=italic_run_reconciliation(items)
        self.assertEqual(items[0]["font_variant"],"italic")
        self.assertTrue(items[0]["italic_run_promoted"])
        self.assertEqual(items[0]["style_source"],"adjacent_italic_run")
        self.assertEqual(promoted,["g0"])

    def test_bubble_without_any_slant_measurement_stays_roman(self):
        # 35 dari 46 tetangga 'tanpa ukuran' memang tegak -> jangan diwarisi.
        items=self._run([(0,"regular",0.0,"dialogue"),(52,"italic",0.455,"dialogue")])
        promoted=italic_run_reconciliation(items)
        self.assertEqual(items[0]["font_variant"],"regular")
        self.assertNotIn("italic_run_promoted",items[0])
        self.assertEqual(promoted,[])

    def test_run_with_no_italic_anchor_is_left_alone(self):
        items=self._run([(0,"regular",0.667,"dialogue"),(52,"regular",0.455,"dialogue")])
        self.assertEqual(italic_run_reconciliation(items),[])
        self.assertEqual([i["font_variant"] for i in items],["regular","regular"])

    def test_neighbour_too_far_or_other_style_is_a_separate_run(self):
        far=self._run([(0,"regular",0.667,"dialogue"),(400,"italic",0.455,"dialogue")])
        other=self._run([(0,"regular",0.667,"dialogue"),(52,"italic",0.455,"narration")])
        self.assertEqual(italic_run_reconciliation(far),[])
        self.assertEqual(italic_run_reconciliation(other),[])
        self.assertEqual(far[0]["font_variant"],"regular")
        self.assertEqual(other[0]["font_variant"],"regular")

    def test_bold_neighbour_keeps_its_weight(self):
        items=self._run([(0,"bold",0.5,"dialogue"),(52,"italic",0.455,"dialogue")])
        italic_run_reconciliation(items)
        self.assertEqual(items[0]["font_variant"],"bold_italic")

    def test_items_without_bbox_are_ignored(self):
        self.assertEqual(italic_run_reconciliation([{"font_variant":"regular"}]),[])


if __name__=='__main__':unittest.main()



class RunUniformityTests(unittest.TestCase):
    """Satu alur bubble harus memakai satu muka huruf."""

    @staticmethod
    def run2(specs):
        """specs: (y, font_variant, raw_italic_score, height) pada gaya sama."""
        return [{"id": f"u{y}", "bbox": {"x": 0, "y": y, "width": 80, "height": h},
                 "surface_type": "BUBBLE_FLAT", "semantic_style": "dialogue",
                 "font_variant": variant, "raw_italic_score": score}
                for y, variant, score, h in specs]

    def test_regular_and_bold_neighbour_settle_on_one_face(self):
        # page-007: "ILLEGAL DUMPING..." (regular) + "OR A FINE OF" (bold) --
        # satu muka huruf di sumber, jatuh di sisi ambang bold yang berbeda.
        items = self.run2([(0, "regular", 0.0, 40), (52, "bold", 0.0, 40)])
        italic_run_reconciliation(items)
        self.assertEqual([i["font_variant"] for i in items], ["regular", "regular"])
        self.assertTrue(any("run_variant_uniformed" in i for i in items))

    def test_strict_bold_majority_keeps_the_whole_run_bold(self):
        items = self.run2([(0, "bold", 0.0, 40), (52, "bold", 0.0, 40), (104, "regular", 0.0, 40)])
        italic_run_reconciliation(items)
        self.assertEqual([i["font_variant"] for i in items], ["bold", "bold", "bold"])

    def test_weak_italic_in_a_mixed_run_is_dropped(self):
        # page-014: skor 0.222 terlalu tipis untuk mempertahankan miring.
        items = self.run2([(0, "italic", 0.222, 40), (52, "regular", 0.0, 40)])
        italic_run_reconciliation(items)
        self.assertEqual([i["font_variant"] for i in items], ["regular", "regular"])

    def test_strong_italic_is_left_alone_beside_a_roman_neighbour(self):
        # bukti miring kuat tidak dihapus hanya karena tetangganya tegak
        items = self.run2([(0, "italic", 0.5, 40), (52, "regular", 0.0, 40)])
        touched = italic_run_reconciliation(items)
        self.assertEqual([i["font_variant"] for i in items], ["italic", "regular"])
        self.assertEqual(touched, [])

    def test_a_short_line_still_joins_the_run_of_its_sentence(self):
        # baris 32 px tak boleh memutus alur: toleransi pakai tinggi terbesar
        items = self.run2([(0, "regular", 0.0, 32), (60, "bold", 0.0, 160)])
        italic_run_reconciliation(items)
        self.assertEqual([i["font_variant"] for i in items], ["regular", "regular"])

    def test_emphasis_run_keeps_its_own_weights(self):
        # alur yang sudah italic semua: bobot bold tiap anggota dipertahankan
        items = self.run2([(0, "bold_italic", 0.5, 40), (52, "italic", 0.5, 40)])
        italic_run_reconciliation(items)
        self.assertEqual([i["font_variant"] for i in items], ["bold_italic", "italic"])
