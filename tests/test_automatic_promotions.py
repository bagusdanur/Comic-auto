import unittest
import numpy as np
from automatic_promotions import detect_promotions,remove_promotions

def r(i,text,y):return dict(id=i,text=text,bbox=dict(x=100,y=y,width=500,height=30))
def banner():return [r('a','YOU CAN FIND-ALL CHAPTERS',500),r('b','EXCLUSIVELY ON OUR WEBSITE:',535),r('c','EN-THUNDERSCANS.COM',570)]

class AutomaticPromotionTests(unittest.TestCase):
    def test_banner(self):self.assertEqual(detect_promotions(banner(),800)[0]['kind'],'PROMOTIONAL_BANNER')
    def test_bubble(self):
        rs=[r('a','THIS CHAPTER IS BROUGHT TO YOU',500),r('b','BY EN-THUNDERSCANS.COM, PLEASE SUPPORT',535),r('c','READING FROM PIRATED WEBSITES',570),r('d',"TO ENSURE IT'S CONTINUITY.",605)]
        self.assertEqual(detect_promotions(rs,800)[0]['kind'],'PROMOTIONAL_BALLOON')
    def test_domain_required(self):self.assertEqual(detect_promotions(banner()[:2],800),[])
    def test_dialogue_and_domain_not_enough(self):self.assertEqual(detect_promotions([r('a','Visit the website tomorrow.',500),banner()[-1]],800),[])
    def test_unrelated_dialogue_blocks_removal(self):self.assertEqual(detect_promotions(banner()+[r('d','I WILL SAVE YOU!',550)],800),[])
    def test_distant_domain_rejected(self):
        rs=banner();rs[-1]['bbox']['y']=1500;self.assertEqual(detect_promotions(rs,800),[])
    def test_inpaint_failure_preserves_output(self):
        a=np.full((1400,800,3),200,np.uint8)
        def fail(a,m):raise RuntimeError('offline')
        result,ids,decisions=remove_promotions(a,a,banner(),[],fail)
        self.assertTrue(np.array_equal(a,result));self.assertFalse(ids);self.assertEqual(decisions[0]['action'],'PRESERVED_PROMOTION')
    def test_mixed_dialogue_group_is_not_removed(self):
        a=np.full((1400,800,3),200,np.uint8)
        result,ids,d=remove_promotions(a,a,banner(),[dict(id='g',source_region_ids=['a','other-dialogue'])])
        self.assertFalse(ids);self.assertTrue(np.array_equal(a,result))
    def test_unsafe_inpaint_rejected(self):
        a=np.full((1400,800,3),200,np.uint8)
        result,ids,d=remove_promotions(a,a,banner(),[],lambda a,m:np.zeros_like(a))
        self.assertTrue(np.array_equal(a,result));self.assertEqual(d[0]['reason'],'UNSAFE_INPAINT_RESULT')
    def test_success_suppresses_group_and_changes_only_local_mask(self):
        a=np.full((1400,800,3),200,np.uint8)
        def clean(a,m):
            b=a.copy();b[m>0]=255;return b
        result,ids,d=remove_promotions(a,a,banner(),[dict(id='g',source_region_ids=['a','b'])],clean)
        self.assertEqual(ids,{'g'});self.assertTrue(np.array_equal(result[:480],a[:480]));self.assertEqual(d[0]['action'],'REMOVED_PROMOTION')
