import unittest
import numpy as np
import cv2
from open_bubble import open_bubble_mask
from outside_comment import comment_candidates

class OpenBubbleTests(unittest.TestCase):
    def scene(self):
        a=np.full((600,600,3),255,np.uint8)
        cv2.ellipse(a,(300,330),(150,100),0,15,345,(0,0,0),3)
        return a,{'x':250,'y':300,'width':100,'height':50}
    def test_partial_oval_detected(self):
        a,b=self.scene();self.assertIsNotNone(open_bubble_mask(a,b,[]))
    def test_blank_area_not_invented(self):
        a,b=self.scene();a[:]=255;self.assertIsNone(open_bubble_mask(a,b,[]))
    def test_single_line_not_balloon(self):
        a,b=self.scene();a[:]=255;cv2.line(a,(100,240),(500,240),(0,0,0),3)
        self.assertIsNone(open_bubble_mask(a,b,[]))
    def test_interior_dialogue_not_relocated(self):
        a,b=self.scene();self.assertEqual(comment_candidates(a,a,b,[{'x':220,'y':180,'width':120,'height':40}]),[])
