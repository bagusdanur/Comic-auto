import unittest
from branding_v51 import confirmed_credit
from branding_v52 import promo_groups
from qimanga_cover import cover_qimanga,is_qimanga_group
from PIL import Image

class QimangaTests(unittest.TestCase):
    def test_inpainted_cover_has_no_backing_and_is_bounded(self):
        image=Image.new('RGB',(800,500),'red')
        logo=Image.new('RGBA',(100,40),(0,0,0,0))
        group=[{'text':'MA.COM','bbox':{'x':669,'y':281,'width':101,'height':38}}]
        def repair(context,mask):
            result=context.copy();result[mask>0]=(10,100,150);return result
        result,(a,b,c,d)=cover_qimanga(image,group,logo,0,500,inpaint=repair)
        self.assertEqual(result.getpixel((a,b)),(10,100,150))
        self.assertEqual(result.getpixel((c-1,d-1)),(10,100,150))
        self.assertEqual(result.getpixel((a-1,b)),(255,0,0))
        self.assertTrue(is_qimanga_group(group,'onimusha/original'))
        self.assertFalse(is_qimanga_group(group,'other-title/original'))
    def test_unsafe_inpaint_rejected(self):
        image=Image.new('RGB',(800,500),'red');logo=Image.new('RGBA',(100,40))
        group=[{'text':'QIMANGA.COM','bbox':{'x':600,'y':200,'width':150,'height':30}}]
        def unsafe(context,mask):return context*0
        with self.assertRaisesRegex(RuntimeError,'outside'):
            cover_qimanga(image,group,logo,0,500,inpaint=unsafe)
    def test_domain_ocr_aliases(self):
        for value in ['qimanga.com','CIMANGA.CON','CIMANEA.COM','QIMANGA.COM.']:
            self.assertTrue(confirmed_credit(value))
        for value in ['CIMANGA','CIMANGA CON','I read CIMANGA.CON yesterday','MANGA.CON']:
            self.assertFalse(confirmed_credit(value))
    def test_overlapping_aliases_are_one_group(self):
        box={'x':616,'y':5472,'width':157,'height':39}
        rs=[{'text':'CIMANGA.CON','bbox':dict(box)},{'text':'CIMANEA.COM','bbox':{**box,'height':38}},{'text':'WHAT?','bbox':dict(box)}]
        gs=promo_groups(rs,800,0,15000)
        self.assertEqual(len(gs),1);self.assertEqual(len(gs[0]),2)
        self.assertNotIn(rs[2],gs[0])
