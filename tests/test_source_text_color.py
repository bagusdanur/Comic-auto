import unittest
import cv2
import numpy as np
from source_text_color import source_text_color


class SourceTextColorTests(unittest.TestCase):
    def sample(self,color,background=(255,255,255),outline=None):
        clean=np.full((100,320,3),background,np.uint8);raw=clean.copy()
        if outline is not None:cv2.putText(raw,'RAW TEXT',(12,65),cv2.FONT_HERSHEY_SIMPLEX,1.3,outline,7,cv2.LINE_AA)
        cv2.putText(raw,'RAW TEXT',(12,65),cv2.FONT_HERSHEY_SIMPLEX,1.3,color,3,cv2.LINE_AA)
        return source_text_color(raw,clean,{'x':5,'y':20,'width':300,'height':60})

    def test_red_fill_not_black_outline(self):
        style=self.sample((230,20,30),outline=(0,0,0))
        self.assertEqual(style['color_source'],'RAW_GLYPH_COLOR')
        self.assertGreater(style['color'][0],200);self.assertLess(style['color'][1],60)

    def test_blue_and_yellow(self):
        for color in ((20,50,230),(245,210,10)):
            result=self.sample(color,background=(40,40,40))
            self.assertTrue(np.max(np.abs(np.array(result['color'])-color))<20)

    def test_red_background_does_not_make_black_text_red(self):
        result=self.sample((0,0,0),background=(220,20,20))
        self.assertEqual(result['color'],(0,0,0))

    def test_white_text_on_light_art_keeps_white(self):
        result=self.sample((255,255,255),background=(160,170,180))
        self.assertEqual(result['color'],(255,255,255))

    def test_near_tied_black_core_and_white_outline_keeps_black_fill(self):
        clean=np.full((100,320,3),(150,170,190),np.uint8);raw=clean.copy()
        cv2.putText(raw,'RAW TEXT',(12,65),cv2.FONT_HERSHEY_SIMPLEX,1.3,(255,255,255),6,cv2.LINE_AA)
        cv2.putText(raw,'RAW TEXT',(12,65),cv2.FONT_HERSHEY_SIMPLEX,1.3,(0,0,0),4,cv2.LINE_AA)
        result=source_text_color(raw,clean,{'x':5,'y':20,'width':300,'height':60})
        self.assertEqual(result['color'],(0,0,0))

    def test_unchanged_colored_background_is_not_a_glyph(self):
        raw=np.full((50,100,3),(230,0,0),np.uint8)
        result=source_text_color(raw,raw.copy(),{'x':0,'y':0,'width':100,'height':50})
        self.assertEqual(result['color_source'],'BACKGROUND_CONTRAST_FALLBACK')

    def test_strong_vertical_raw_gradient_uses_solid_color(self):
        clean=np.full((120,240,3),25,np.uint8);raw=clean.copy()
        mask=np.zeros((120,240),np.uint8)
        cv2.putText(mask,'WOW',(15,85),cv2.FONT_HERSHEY_SIMPLEX,2.2,255,8,cv2.LINE_AA)
        for y in range(120):
            color=np.array((250-round(y*1.3),40,30+round(y*1.5))).clip(0,255)
            raw[y][mask[y]>0]=color
        result=source_text_color(raw,clean,{'x':0,'y':0,'width':240,'height':120})
        self.assertEqual(result['fill_mode'],'solid')
