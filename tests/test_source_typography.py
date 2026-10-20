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
