import unittest
from unittest.mock import patch
import numpy as np
from vortex_emblems import find_emblems,clean_emblems

class EmblemTests(unittest.TestCase):
    def test_blank_not_emblem(self):
        template=np.zeros((95,88),np.uint8);template[20:70,20:60]=255
        self.assertEqual(find_emblems(np.full((300,800,3),255,np.uint8),template),[])
    def test_confirmed_match(self):
        rng=np.random.default_rng(4);template=(rng.random((95,88))>.5).astype(np.uint8)*255
        image=np.zeros((300,800,3),np.uint8);image[100:195,200:288,0]=template
        found=find_emblems(image,template)
        self.assertTrue(any(abs(x-200)<2 and abs(y-100)<2 for score,x,y,w,h in found))
