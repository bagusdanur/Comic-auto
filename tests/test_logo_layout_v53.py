import unittest
import numpy as np
import cv2
from PIL import Image
from logo_layout_v53 import fit_larger_logo


class LogoLayoutV53Tests(unittest.TestCase):
    def test_larger_than_source_height_limit(self):
        image=np.full((400,800,3),255,np.uint8);mask=np.zeros((400,800),np.uint8);mask[170:225,100:340]=255
        logo=Image.new('RGBA',(220,100),'purple')
        mark,pos,report=fit_larger_logo(logo,(100,170,340,225),mask,image,[])
        self.assertGreater(mark.width,160)
        self.assertEqual(report['placement_engine'],'v5.3')

    def test_nearby_dialogue_not_covered(self):
        image=np.full((400,800,3),255,np.uint8);mask=np.zeros((400,800),np.uint8);mask[170:225,100:340]=255
        protected=[{'bbox':{'x':100,'y':120,'width':240,'height':40}}]
        mark,(x,y),_=fit_larger_logo(Image.new('RGBA',(220,100),'purple'),(100,170,340,225),mask,image,protected)
        self.assertGreaterEqual(y,165)

    def test_no_space_fails(self):
        image=np.full((400,800,3),255,np.uint8);mask=np.zeros((400,800),np.uint8)
        with self.assertRaisesRegex(RuntimeError,'safely'):
            fit_larger_logo(Image.new('RGBA',(220,100),'purple'),(100,170,340,225),mask,image,[{'bbox':{'x':0,'y':0,'width':800,'height':400}}])

    def test_enlargement_avoids_long_panel_border(self):
        image=np.full((400,800,3),255,np.uint8)
        cv2.line(image,(40,230),(700,230),(0,0,0),2)
        mask=np.zeros((400,800),np.uint8);mask[170:225,100:340]=255
        mark,(x,y),_=fit_larger_logo(Image.new('RGBA',(220,100),'purple'),(100,170,340,225),mask,image,[])
        self.assertGreater(mark.width,160)
        self.assertTrue(y+mark.height<229 or y>231)
