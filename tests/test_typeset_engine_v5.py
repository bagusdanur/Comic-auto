import unittest
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typeset_engine_v5 import fit_text_v5, fit_constrained_text, symmetric_outline_mask, paint_fitted_text


class V5GeometryTests(unittest.TestCase):
    font = 'assets/fonts/wildwords/CC Wild Words Roman.ttf'

    def test_outline_expands_fill_symmetrically(self):
        fill=Image.new('L',(30,30),0)
        ImageDraw.Draw(fill).rectangle((10,10,19,19),fill=255)
        outlined=np.asarray(symmetric_outline_mask(fill,2))>0
        yy,xx=np.where(outlined)
        self.assertEqual((xx.min(),yy.min(),xx.max(),yy.max()),(8,8,21,21))

    def test_rotated_text_and_outline_share_one_safe_mask(self):
        output=Image.new('RGB',(180,100),'white')
        font=ImageFont.truetype(self.font,24)
        fitted={'origin':(35,28),'wrapped':'MIRING','font':font,'spacing':3}
        safe=np.ones((100,180),np.uint8)*255
        self.assertTrue(paint_fitted_text(
            output,(0,0,180,100),fitted,{'color':(255,255,255),'fill_mode':'solid'},
            2,(0,0,0),rotation=12,rotation_center=(90,50),safe_mask=safe,
        ))
        pixels=np.asarray(output)
        self.assertGreater(np.count_nonzero(np.any(pixels<250,axis=2)),0)

    def check_fit(self, mask):
        text = 'AKU AKAN KEMBALI BERSAMA TEMAN-TEMANKU.'
        fitted = fit_text_v5(text, self.font, mask, 22, 36)
        self.assertIsNotNone(fitted)
        self.assertEqual(' '.join(fitted['wrapped'].split()), text)
        layer = Image.new('L', (mask.shape[1], mask.shape[0]))
        ImageDraw.Draw(layer).multiline_text(fitted['origin'], fitted['wrapped'], font=fitted['font'], spacing=fitted['spacing'], align='center', fill=255)
        self.assertFalse(np.any((np.array(layer)>0) & (mask==0)))
        self.assertGreaterEqual(fitted['font_size'],22)

    def test_oval(self):
        mask=np.zeros((210,340),np.uint8)
        cv2.ellipse(mask,(170,105),(155,95),0,0,360,255,-1)
        self.check_fit(mask)

    def test_long_indonesian_word_in_narrow_oval(self):
        mask=np.zeros((196,182),np.uint8)
        cv2.ellipse(mask,(91,98),(65,90),0,0,360,255,-1)
        result=fit_constrained_text('Aku bisa mengatasinya.',self.font,mask)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result['font_size'],18)
        restored=result['wrapped'].replace('-\n','').replace('\n',' ')
        self.assertEqual(restored,'Aku bisa mengatasinya.')
        layer=Image.new('L',(182,196))
        ImageDraw.Draw(layer).multiline_text(result['origin'],result['wrapped'],font=result['font'],spacing=result['spacing'],align='center',fill=255)
        self.assertFalse(np.any((np.array(layer)>0)&(mask==0)))

    def test_constrained_still_rejects_impossible_area(self):
        self.assertIsNone(fit_constrained_text('Mengatasinya.',self.font,np.ones((12,20),np.uint8)*255))

    def test_spiky(self):
        mask=np.zeros((210,340),np.uint8)
        points=np.array([[5,95],[50,75],[30,20],[105,45],[170,5],[195,40],[315,20],[285,70],[335,110],[290,140],[310,195],[220,165],[160,205],[115,170],[20,195],[55,140]])
        cv2.fillPoly(mask,[points],255)
        self.check_fit(mask)

    def test_rectangle(self):
        self.check_fit(np.pad(np.ones((180,310),np.uint8)*255,15))

    def test_impossible_does_not_clip_or_shrink_below_floor(self):
        self.assertIsNone(fit_text_v5('TERLALU PANJANG UNTUK AREA INI',self.font,np.ones((15,30),np.uint8)*255,22,36))

    def test_short_text_prefers_maximum(self):
        result=fit_text_v5('JANGAN!',self.font,np.ones((160,300),np.uint8)*255,22,36)
        self.assertEqual(result['font_size'],36)

    def test_artwork_outline_is_inside_mask(self):
        mask=np.pad(np.ones((140,270),np.uint8)*255,10)
        result=fit_text_v5('JANGAN PERGI!',self.font,mask,22,36,stroke=2)
        self.assertIsNotNone(result)
        layer=Image.new('L',(mask.shape[1],mask.shape[0]))
        ImageDraw.Draw(layer).multiline_text(result['origin'],result['wrapped'],font=result['font'],spacing=result['spacing'],align='center',fill=255,stroke_width=2,stroke_fill=255)
        self.assertFalse(np.any((np.array(layer)>0)&(mask==0)))

    def test_asymmetric_bubble_keeps_raw_text_anchor(self):
        mask=np.ones((180,360),np.uint8)*255
        fitted=fit_text_v5('TETAP DI SINI',self.font,mask,22,36,anchor_center=(115,78))
        self.assertIsNotNone(fitted)
        box=ImageDraw.Draw(Image.new('L',(1,1))).multiline_textbbox(
            fitted['origin'],fitted['wrapped'],font=fitted['font'],spacing=fitted['spacing'],align='center'
        )
        center=((box[0]+box[2])/2,(box[1]+box[3])/2)
        self.assertLess(abs(center[0]-115),12)
        self.assertLess(abs(center[1]-78),12)
