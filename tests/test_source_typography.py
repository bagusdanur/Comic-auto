import unittest
from types import SimpleNamespace
import numpy as np
from source_typography import source_typography,select_font


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


if __name__=='__main__':unittest.main()
