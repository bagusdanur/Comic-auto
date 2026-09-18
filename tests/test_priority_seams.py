import unittest,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from priority_seams import boundary_offset,prepare_inputs

class SeamTests(unittest.TestCase):
    def test_move_to_white_gutter(self):
        a=np.zeros((2000,80,3),np.uint8);b=a.copy();b[200:240]=255
        self.assertTrue(200<=boundary_offset(a,b)<=240)
    def test_no_gutter_keeps_boundary(self):
        a=np.zeros((2000,80,3),np.uint8);self.assertEqual(boundary_offset(a,a),0)
    def test_different_width_unchanged(self):self.assertEqual(boundary_offset(np.zeros((500,50,3)),np.zeros((500,60,3))),0)
    def test_lossless_resegmentation_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);a=np.zeros((2000,80,3),np.uint8);b=np.full_like(a,100);b[200:240]=255
            Image.fromarray(a).save(p/'a.png');Image.fromarray(b).save(p/'b.png')
            paths=prepare_inputs([p/'a.png',p/'b.png'],p/'prepared')
            rebuilt=np.concatenate([np.array(Image.open(x)) for x in paths])
            self.assertTrue(np.array_equal(rebuilt,np.concatenate((a,b))))
            self.assertEqual(paths,prepare_inputs([p/'a.png',p/'b.png'],p/'prepared'))
