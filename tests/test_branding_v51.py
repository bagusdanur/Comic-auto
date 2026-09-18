import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from branding_v51 import confirmed_credit,clean_credit,gutter_position,apply_branding_v51


class BrandingV51Tests(unittest.TestCase):
    def test_dialogue_keywords_are_not_logos(self):
        for text in ('SCAN THE ENEMY','READ THIS LETTER','SUPPORT OUR FRIEND','OUR WEBSITE IS BROKEN'):
            self.assertFalse(confirmed_credit(text))
        self.assertTrue(confirmed_credit('READ AT EN-THUNDERSCANS.COM'))

    def test_evascans_source_variants_are_confirmed(self):
        for text in ('EVASCANS', 'EVA SCANS', 'EVASCANS.ORG', 'EVASCANS.0RC'):
            self.assertTrue(confirmed_credit(text), text)
        self.assertFalse(confirmed_credit('Eva scans the room carefully'))

    def test_isolated_mark_only_changes_local_foreground(self):
        im=Image.new('RGBA',(400,300),'white')
        ImageDraw.Draw(im).rectangle((110,110,170,130),fill='black')
        before=np.array(im)
        result=clean_credit(im,{'text':'SOURCE.COM','bbox':{'x':105,'y':105,'width':70,'height':30}},[])
        self.assertEqual(result['action'],'CLEANED')
        changed=np.any(np.array(im)!=before,axis=2)
        self.assertTrue(np.any(changed))
        self.assertFalse(np.any(changed[:100]))
        self.assertFalse(np.any(changed[140:]))

    def test_artwork_is_not_erased(self):
        im=Image.fromarray(np.random.default_rng(1).integers(0,255,(300,400,3),dtype=np.uint8)).convert('RGBA')
        before=np.array(im)
        r=clean_credit(im,{'text':'SOURCE.COM','bbox':{'x':105,'y':105,'width':70,'height':30}},[])
        self.assertEqual(r['action'],'PRESERVED')
        np.testing.assert_array_equal(im,before)

    def test_dialogue_overlap_is_not_erased(self):
        im=Image.new('RGBA',(400,300),'white')
        box={'x':100,'y':100,'width':80,'height':30}
        r=clean_credit(im,{'text':'SOURCE.COM','bbox':box},[{'bbox':box}])
        self.assertEqual(r['reason'],'NEAR_PROTECTED_TEXT_OR_SURFACE')

    def test_white_bubble_in_art_is_not_a_gutter(self):
        im=Image.new('RGBA',(400,300),'black')
        ImageDraw.Draw(im).ellipse((30,30,370,270),fill='white')
        self.assertIsNone(gutter_position(im,Image.new('RGBA',(70,30)),[]))

    def test_logo_uses_full_width_blank_gutter(self):
        im=Image.new('RGBA',(400,400),'black')
        ImageDraw.Draw(im).rectangle((0,150,399,250),fill='white')
        p=gutter_position(im,Image.new('RGBA',(70,30)),[])
        self.assertIsNotNone(p)
        self.assertGreaterEqual(p[1],150)
        self.assertLessEqual(p[1]+30,250)

    def test_footer_fallback_preserves_all_original_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);im=Image.new('RGB',(400,300),(70,80,90));im.save(root/'page.png')
            (root/'ocr.json').write_text(json.dumps({'regions':[]}))
            Image.new('RGBA',(100,40),(90,10,200,255)).save(root/'logo.png')
            result=apply_branding_v51(root/'page.png',root/'ocr.json',root/'logo.png',root/'unused.png',first_page=True)
            self.assertEqual(result['branding_engine'],'v5.1')
            self.assertEqual(result['replacements'][0]['mode'],'ADDED_BRANDING_STRIP')
            with Image.open(root/'page.png') as after:
                np.testing.assert_array_equal(np.array(after)[:300],np.array(im))
                self.assertGreater(after.height,300)
