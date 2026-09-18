import unittest
import numpy as np
from branding_ink_guard import measured_ink_region,intersects_protected


class InkGuardTests(unittest.TestCase):
    def setUp(self):
        self.clean=np.full((160,240,3),255,np.uint8)
        self.render=self.clean.copy();self.render[50:65,110:150]=0
        self.item={'id':'dialogue','bbox':{'x':70,'y':20,'width':150,'height':110}}

    def test_empty_layout_corner_is_not_dialogue(self):
        region=measured_ink_region(self.item,self.render,self.clean)
        logo=np.ones((15,30),np.uint8)
        self.assertFalse(intersects_protected(logo,(75,105),region))

    def test_real_glyph_overlap_is_blocked(self):
        region=measured_ink_region(self.item,self.render,self.clean)
        self.assertTrue(intersects_protected(np.ones((15,30),np.uint8),(115,52),region))

    def test_missing_cleaned_image_keeps_box_protection(self):
        region=measured_ink_region(self.item,self.render,None)
        self.assertTrue(intersects_protected(np.ones((15,30),np.uint8),(75,105),region))

    def test_no_detectable_ink_keeps_box_protection(self):
        region=measured_ink_region(self.item,self.clean,self.clean)
        self.assertNotIn('_ink_mask',region)

    def test_ink_padding_protects_antialias_edges(self):
        region=measured_ink_region(self.item,self.render,self.clean)
        self.assertTrue(intersects_protected(np.ones((1,1),np.uint8),(109,49),region))
