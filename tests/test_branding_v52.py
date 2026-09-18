import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageDraw
from branding_v52 import card_bounds,pill_mask,eva_pill_mask,dusk_pill_mask,promo_groups,text_mask,apply_branding_v52,standalone_promo,standalone_madara_promo,standalone_eva_promo,short_vortex_end_promo,promo_text


class MadaraPromoTests(unittest.TestCase):
    def test_dusk_mask_covers_avatar_and_capsule(self):
        image=Image.new('RGB',(800,300),(90,90,90));draw=ImageDraw.Draw(image)
        draw.ellipse((4,53,118,159),fill=(65,5,5),outline=(220,20,20),width=8)
        draw.rectangle((90,70,300,145),fill=(25,10,10),outline=(210,20,20),width=5)
        group=[{'text':'DUSKSCANS.COM','bbox':{'x':109,'y':97,'width':183,'height':41}}]
        mask,bounds=dusk_pill_mask(np.array(image),group)
        self.assertEqual(bounds,(0,49,312,168))
        self.assertTrue(mask[100,10])
        self.assertTrue(mask[100,290])

    def test_short_madara_card_is_source_promo(self):
        page=np.zeros((517,1027,3),dtype=np.uint8)
        regions=[
            {'text':'YOU CAN READ','bbox':{'x':188,'y':37,'width':310,'height':34}},
            {'text':'CHAPTERS','bbox':{'x':47,'y':64,'width':554,'height':153}},
            {'text':'FROM','bbox':{'x':102,'y':238,'width':112,'height':61}},
            {'text':'MADARASCANS.COM','bbox':{'x':227,'y':381,'width':403,'height':67}},
        ]
        self.assertTrue(standalone_madara_promo(page,regions))

    def test_full_eva_end_card_is_source_promo(self):
        page=np.zeros((1058,690,3),dtype=np.uint8)
        page[:,:,0]=180;page[:,:,1]=30;page[:,:,2]=55
        regions=[
            {'text':'READ ONLY AT EVA SCANS','bbox':{'x':60,'y':60,'width':300,'height':120}},
            {'text':'PLEASE SUPPORT US BY READING THIS AT OUR SITE','bbox':{'x':60,'y':450,'width':280,'height':180}},
            {'text':'REACH OUT!','bbox':{'x':150,'y':760,'width':240,'height':80}},
            {'text':'DISCORD.GG/example EVASCANS.ORG','bbox':{'x':200,'y':870,'width':400,'height':140}},
        ]
        self.assertTrue(standalone_eva_promo(page,regions))

    def test_compact_madara_domain_uses_full_pill_mask(self):
        # Regression fixture: domain OCR begins right of a disconnected round
        # emblem. The returned mask must include pixels left of the text box.
        page=np.full((300,800,3),245,dtype=np.uint8)
        cv2.circle(page,(55,130),28,(110,15,150),-1)
        cv2.rectangle(page,(80,110),(255,150),(25,10,30),-1)
        cv2.rectangle(page,(78,106),(260,154),(120,15,155),3)
        group=[{'text':'MADARASCANS.COM','bbox':{'x':92,'y':118,'width':150,'height':24}}]
        mask,bounds=pill_mask(page,group)
        self.assertLessEqual(bounds[0],30)
        self.assertGreater(mask[130,55],0)
from branding_coordinates import project_region


def region(text,x,y,w=120,h=15):
    return {'text':text,'bbox':{'x':x,'y':y,'width':w,'height':h}}


class BrandingV52Tests(unittest.TestCase):
    def test_tiny_thunder_caption_ocr_alias_is_promo(self):
        self.assertTrue(promo_text('COU CAN READ THE CHAPTER ON'))
        self.assertTrue(promo_text('YOU CAN READ THE CHAPTER ON'))
        self.assertTrue(promo_text('APTER ON'))
        self.assertTrue(promo_text('YOU CANREAD THE CHAPTER ON'))
        self.assertFalse(promo_text('YOU CAN UNDERSTAND MY MIND'))

    def test_text_watermark_is_added_to_safe_white_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            Image.new('RGB',(400,500),'white').save(p/'page.png')
            Image.new('RGBA',(100,40),(128,0,128,255)).save(p/'logo.png')
            Image.new('RGBA',(200,30),(70,20,150,255)).save(p/'watermark-text.png')
            (p/'ocr.json').write_text(json.dumps({'regions':[]}))
            report=apply_branding_v52(p/'page.png',p/'ocr.json',p/'logo.png',p/'watermark-text.png',inpaint=lambda a,m:a)
            self.assertIsNotNone(report['text_watermark'])
            self.assertTrue(np.any(np.array(Image.open(p/'page.png'))!=255))

    def test_vortex_ocr_alias_is_standalone_only(self):
        from branding_v51 import confirmed_credit
        self.assertTrue(confirmed_credit('VURTEXSCANS.CUM'))
        self.assertFalse(confirmed_credit('I saw VURTEXSCANS.CUM yesterday'))
    def test_vortex_illustrated_card_requires_anchors(self):
        a=np.full((1500,800,3),50,np.uint8);a[481:700]=255
        rs=[region('VORTEX',161,15),region('SCANS',170,72),region('THIS CHAPTER WAS BROUGHT TO',56,209),region('YOU BY THE VORTEX SCANS TEAM',49,255),region('SHOW YOUR SUPPORT AND JOIN THE',33,298),region('STORM AT VORTEXSCANS.ORG',67,348),region('VORTEXSCANS.ORG',531,448,214,21)]
        self.assertEqual(card_bounds(a,rs)[0],481)
        self.assertEqual(card_bounds(a,rs+[region('This is actual story dialogue',50,400)])[0],0)
    def test_optional_branding_400_preserves_pixels(self):
        from requests import Response
        from requests.exceptions import HTTPError
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);Image.new('RGB',(400,500),'white').save(p/'page.png')
            Image.new('RGBA',(100,50),'purple').save(p/'logo.png')
            (p/'ocr.json').write_text(json.dumps({'regions':[region('READ AT SOURCE.COM',50,200,200,20)]}))
            def rejected(a,m):
                response=Response();response.status_code=400;response._content=b'{"detail":"mask exceeds page safety limit"}'
                raise HTTPError(response=response)
            report=apply_branding_v52(p/'page.png',p/'ocr.json',p/'logo.png',None,inpaint=rejected)
            self.assertTrue(any(d['action']=='PRESERVED_REJECTED_SOURCE_MASK' for d in report['decisions']))
            self.assertTrue(np.all(np.array(Image.open(p/'page.png'))==255))

    def test_vortex_ads_preserve_story_signoff(self):
        a=np.full((1420,800,3),45,np.uint8);a[940:1040]=255;a[1040:]=0
        rs=[region('READ THIS SERIES FIRST AT:',29,51,742,147),region('VORTEXSCANS.COM',3,161,790,203),region('To be continued',498,624,262,32),region('DEAR READERS',319,1083),region('DIVE DEEPER INTO THE STORY',217,1145),region('UNLOCK UP TO THE LATEST CHAPTER',206,1190),region('EXCLUSIVELY ON VORTEXSCANS.ORG!',204,1237),region('HAPPY-READING,',312,1308)]
        top,bottom=card_bounds(a,rs)
        self.assertGreater(top,364);self.assertLess(top,624)
        self.assertEqual(bottom,940)
        self.assertEqual(card_bounds(a,rs+[region('Actual dialogue here',100,300)])[0],0)
        self.assertEqual(card_bounds(a,rs+[region('Actual dialogue here',100,1250)])[1],1420)

    def test_short_vortex_end_promo_is_replaced(self):
        image=np.zeros((474,800,3),np.uint8);image[:,:,0]=20
        image[:,::8,0]=180
        rs=[region('DEAR READERS DIVE DEEPER INTO THE STORY-AND UNLOCK UP TO THE LATEST CHAPTER',204,134,408,152),region('EXCLUSIVELY ON VORTEXSCANS.ORG!',205,292,390,32),region('HAPPY READING,',312,360,187,49)]
        self.assertTrue(short_vortex_end_promo(image,rs))
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);Image.fromarray(image).save(p/'page.png')
            Image.new('RGBA',(100,50),(128,0,128,255)).save(p/'logo.png')
            (p/'ocr.json').write_text(json.dumps({'regions':rs}))
            report=apply_branding_v52(p/'page.png',p/'ocr.json',p/'logo.png',None,inpaint=lambda a,m:a)
            self.assertTrue(any(d['action']=='REPLACED_SHORT_VORTEX_END_PROMO' for d in report['decisions']))
            after=np.array(Image.open(p/'page.png'))
            self.assertFalse(np.any((after[:,:,0]>120)&(after[:,:,1]<90)&(after[:,:,2]<90)))

    def test_emblem_height_rounds_to_pixel_at_690_width(self):
        im=Image.new('RGB',(690,300),'white');d=ImageDraw.Draw(im)
        d.rectangle((4,80,97,169),fill=(170,0,220))
        d.rectangle((97,100,350,150),fill='black')
        g=[region('YOU CAN READ THE CHAPTER ON:',106,110,231,17),region('EN-THUNDERSCANS.COM',106,127,231,19)]
        mask,bounds=pill_mask(np.array(im),g)
        self.assertGreater(np.count_nonzero(mask),0)
        self.assertLess(bounds[0],10)

    def test_unrecognized_emblem_does_not_abort_chapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);Image.new('RGB',(400,500),'white').save(p/'page.png')
            Image.new('RGBA',(100,50),'purple').save(p/'logo.png')
            rs=[region('YOU CAN READ THE CHAPTER ON:',80,200,180,15),region('EN-THUNDERSCANS.COM',80,215,180,15)]
            (p/'ocr.json').write_text(json.dumps({'regions':rs}))
            report=apply_branding_v52(p/'page.png',p/'ocr.json',p/'logo.png',None,inpaint=lambda a,m:a)
            self.assertTrue(any(d['action']=='PRESERVED_UNCERTAIN_SOURCE_MARK' for d in report['decisions']))

    def test_warning_followed_by_black_gutter(self):
        image=np.full((7000,800,3),255,np.uint8);image[:1520]=17;image[1520:1900]=0
        rs=[region('WARNING!',40,535),region('READER SUPPORT',30,790),region('OUR WEBSITE',30,840),region('EN-THUNDERSCANS.COM',220,1027),region('QUALITY TRANSLATION',180,1230),region('THUNDERSCANS TEAM',490,1338)]
        self.assertEqual(card_bounds(image,rs),(1520,7000))
        self.assertEqual(card_bounds(image,[region('WARNING!',40,535)]),(0,7000))
    def test_recruitment_crop_preserves_publisher_credit(self):
        a=np.full((2000,800,3),100,np.uint8);a[1710:1730]=0
        rs=[region('CAPCOM',550,1650,180,50),region('WE ARE RECRUITING',420,1770),region('POSITIONS AVAILABLE',450,1800),region('JOIN US TO WORK',450,1850),region('DISCORD.GG/ABC',450,1900)]
        top,bottom=card_bounds(a,rs)
        self.assertEqual(top,0);self.assertGreater(bottom,1700);self.assertLess(bottom,1730)
        self.assertEqual(card_bounds(a,rs[1:]),(0,2000))

    def test_merger_card_requires_anchors_and_dark_gutter(self):
        a=np.full((2000,800,3),100,np.uint8);a[480:700]=0
        rs=[region('OiSCANS',42,20),region('MERGER',137,58),region('ANNOUNCEMENT',131,107),region('LIGHTNING FAST RELEASES',80,250),region('DISCORD.GG/SCANS',80,396)]
        self.assertEqual(card_bounds(a,rs),(480,2000))
        self.assertEqual(card_bounds(a,rs[:-1]),(0,2000))
        a[480:700]=100
        self.assertEqual(card_bounds(a,rs),(0,2000))

    def test_standalone_promo_requires_ad_evidence_and_no_dialogue(self):
        image=np.zeros((778,800,3),np.uint8)
        rs=[region('READ ONLY AT',114,20),region('SCANS',320,234),region('PLEASE SUPPORT US',350,500),region('DISCORD.GG/ABC',430,642),region('EN-THUNDERSCANS.COM',430,712)]
        self.assertTrue(standalone_promo(image,rs))
        self.assertFalse(standalone_promo(image,rs+[region('WHERE HAVE YOU BEEN ALL DAY?',20,300)]))
        self.assertFalse(standalone_promo(image,rs[:-1]))

    def test_inpaint_outside_mask_rejected_before_file_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);image=Image.new('RGB',(400,500),'white');image.save(p/'page.png')
            Image.new('RGBA',(100,50),'purple').save(p/'logo.png')
            (p/'ocr.json').write_text(json.dumps({'regions':[region('READ AT SOURCE.COM',50,200,200,20)]}))
            before=(p/'page.png').read_bytes()
            def unsafe(source,mask):
                out=source.copy();out[0,0]=0;return out
            with self.assertRaisesRegex(RuntimeError,'outside source mask'):
                apply_branding_v52(p/'page.png',p/'ocr.json',p/'logo.png',p/'unused',inpaint=unsafe)
            self.assertEqual((p/'page.png').read_bytes(),before)

    def test_logo_replacement_stays_inside_pill_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);image=Image.new('RGB',(400,300),'white');d=ImageDraw.Draw(image)
            d.polygon([(110,128),(240,128),(258,144),(240,160),(110,160)],fill='black',outline=(170,0,220),width=2)
            d.ellipse((92,122,126,163),fill=(170,0,220));image.save(p/'page.png')
            rs=[region('YOU CAN READ THE CHAPTER ON',135,133,90,10),region('EN-THUNDERSCANS.COM',135,145,90,10)]
            (p/'ocr.json').write_text(json.dumps({'regions':rs}))
            Image.new('RGBA',(100,40),'purple').save(p/'logo.png')
            def safe(source,mask):
                out=source.copy();out[mask>0]=255;return out
            report=apply_branding_v52(p/'page.png',p/'ocr.json',p/'logo.png',p/'unused',inpaint=safe)
            self.assertEqual(report['decisions'][0]['action'],'REPLACED_PILL')
            mask,_=pill_mask(np.array(image),rs)
            after=np.array(Image.open(p/'page.png'))
            self.assertFalse(np.any(np.any(after!=np.array(image),axis=2)&(mask==0)))
    def test_crop_shifts_preview_boxes_without_mutating_raw(self):
        b={'x':10,'y':700,'width':100,'height':40};item={'id':'a','bbox':dict(b)}
        report={'branding_engine':'v5.2','source_header_trimmed_at':500,'source_footer_trimmed_at':900,'input_dimensions':[400,1000]}
        projected=project_region(item,b,report)
        self.assertEqual(projected[0]['y'],200)
        self.assertEqual(projected[1]['bbox']['y'],200)
        self.assertEqual(b['y'],700)
        self.assertIsNone(project_region(item,{**b,'y':950},report))
    def test_warning_card_crops_at_visual_edge(self):
        image=np.full((2000,400,3),255,np.uint8);image[:500]=15
        rs=[region('WARNING',20,30),region('OUR WEBSITE',20,150),region('EN-THUNDERSCANS.COM',20,230),region('SUPPORT US',20,300),region('QUALITY TRANSLATION',20,370),region('THUNDERSCANS TEAM',20,430)]
        self.assertEqual(card_bounds(image,rs),(500,2000))

    def test_warning_without_brand_evidence_does_not_crop(self):
        image=np.full((2000,400,3),255,np.uint8);image[:500]=15
        self.assertEqual(card_bounds(image,[region('WARNING!',20,30)]),(0,2000))

    def test_warning_below_stacked_top_banner(self):
        image=np.full((7000,800,3),255,np.uint8);image[:1500]=15
        rs=[region('YOU CAN READ CHAPTERS',80,150),region('WARNING!',40,535),region('READER SUPPORT',30,790),region('OUR WEBSITE',30,840),region('EN-THUNDERSCANS.COM',220,1027),region('QUALITY TRANSLATION',180,1230),region('THUNDERSCANS TEAM',490,1338)]
        self.assertEqual(card_bounds(image,rs),(1500,7000))

    def test_warning_search_reaches_second_stacked_banner(self):
        image=np.full((7000,800,3),255,np.uint8);image[:1500]=15
        rs=[region('WARNING!',40,130),region('READER SUPPORT',30,390),
            region('OUR WEBSITE',30,440),region('EN-THUNDERSCANS.COM',220,600),
            region('QUALITY TRANSLATION',180,800),region('THUNDERSCANS TEAM',490,930),
            region('YOU CANREAD THE CHAPTER ON:',80,1210),region('EN-THUNDERSCANS.COM',80,1360)]
        self.assertEqual(card_bounds(image,rs),(1500,7000))

    def test_colored_separator_before_white_title_page_is_cropped(self):
        image=np.full((7000,800,3),255,np.uint8)
        image[:1490]=15
        image[1490:1500]=[120,20,180]
        rs=[region('WARNING!',40,130),region('READER SUPPORT',30,390),
            region('OUR WEBSITE',30,440),region('EN-THUNDERSCANS.COM',220,600),
            region('QUALITY TRANSLATION',180,800),region('THUNDERSCANS TEAM',490,930),
            region('YOU CANREAD THE CHAPTER ON:',80,1210),region('EN-THUNDERSCANS.COM',80,1360)]
        self.assertEqual(card_bounds(image,rs),(1490,7000))

    def test_footer_keeps_publisher_credit_above_promo(self):
        image=np.full((1000,400,3),255,np.uint8);image[700:]=15
        rs=[region('D&C MEDIA',20,670),region('READ ONLY AT',20,750),region('DISCORD.GG/SOURCE',20,880),region('EN-THUNDERSCANS.COM',20,940)]
        self.assertEqual(card_bounds(image,rs),(0,700))

    def test_footer_requires_visual_edge(self):
        image=np.full((1000,400,3),130,np.uint8)
        rs=[region('READ ONLY AT',20,750),region('DISCORD.GG/SOURCE',20,880),region('EN-THUNDERSCANS.COM',20,940)]
        self.assertEqual(card_bounds(image,rs),(0,1000))

    def test_support_line_is_grouped_with_read_at(self):
        rs=[region('READ AT EN-THUNDERSCANS.COM',20,100,300,30),region('TO SUPPORT US!',60,135,180,30),region('REAL DIALOGUE',100,210)]
        group=promo_groups(rs,400,0,500)[0]
        self.assertEqual(len(group),2)
        mask=text_mask(np.zeros((500,400,3),np.uint8),group)
        self.assertTrue(mask[150,80])
        self.assertFalse(mask[210,110])

    def test_pill_includes_emblem_left_of_ocr(self):
        im=Image.new('RGB',(400,300),'white');d=ImageDraw.Draw(im)
        d.rectangle((110,130,230,158),fill='black')
        d.ellipse((92,122,126,163),fill=(170,0,220))
        rs=[region('YOU CAN READ THE CHAPTER ON',135,133,90,10),region('EN-THUNDERSCANS.COM',135,145,90,10)]
        mask,box=pill_mask(np.array(im),rs)
        self.assertLess(box[0],100)
        self.assertTrue(mask[135,105])
        self.assertFalse(mask[180,105])

    def test_evascans_pill_includes_portrait_left_of_ocr(self):
        image=np.full((500,800,3),255,np.uint8)
        image[180:244,18:225]=[45,20,35]
        image[180:244,18:65]=[180,35,110]
        group=[region('READ THIS SERIES ONLY AT',92,197,107,13),
               region('EVASCANS.ORG',99,210,95,14)]
        mask,bounds=eva_pill_mask(image,group)
        self.assertLessEqual(bounds[0],18)
        self.assertTrue(mask[200,30])
        self.assertGreaterEqual(bounds[2],225)

    def test_pill_without_emblem_fails_not_guesses(self):
        with self.assertRaisesRegex(RuntimeError,'emblem'):
            pill_mask(np.full((300,400,3),255,np.uint8),[region('EN-THUNDERSCANS.COM',135,145,90,10)])

    def test_thin_purple_art_fragment_is_not_logo_tip(self):
        im=Image.new('RGB',(800,400),'white');d=ImageDraw.Draw(im)
        d.ellipse((0,2,90,87),fill=(170,0,220))
        d.polygon([(128,21),(330,21),(359,48),(330,76),(128,76)],fill='black',outline=(170,0,220),width=3)
        d.rectangle((395,40,415,42),fill=(170,0,220))
        rs=[region('YOU CAN READ THE CHAPTER ON:',103,32,215,14),region('EN-THUNDERSCANS.COM',96,46,224,22)]
        mask,box=pill_mask(np.array(im),rs)
        self.assertLess(box[2],390)
        self.assertFalse(mask[41,405])

    def test_large_purple_art_without_black_interior_is_not_tip(self):
        im=Image.new('RGB',(800,400),'white');d=ImageDraw.Draw(im)
        d.ellipse((0,2,90,87),fill=(170,0,220))
        d.polygon([(128,21),(330,21),(359,48),(330,76),(128,76)],fill='black',outline=(170,0,220),width=3)
        d.rectangle((390,30,415,70),fill=(170,0,220))
        rs=[region('YOU CAN READ THE CHAPTER ON:',103,32,215,14),region('EN-THUNDERSCANS.COM',96,46,224,22)]
        mask,box=pill_mask(np.array(im),rs)
        self.assertLess(box[2],390)
        self.assertFalse(mask[50,400])

    def test_large_confirmed_pill_can_exceed_half_page(self):
        im=Image.new('RGB',(800,400),'white');d=ImageDraw.Draw(im)
        d.polygon([(510,130),(767,130),(799,160),(767,192),(510,192)],fill='black',outline=(170,0,220),width=3)
        d.ellipse((390,110,490,208),fill=(170,0,220))
        rs=[region('YOU CAN READ THE CHAPTER ON:',504,148,250,17),region('EN-THUNDERSCANS.COM',501,168,254,21)]
        mask,box=pill_mask(np.array(im),rs)
        self.assertGreater(box[2]-box[0],400)
        self.assertLess(box[2]-box[0],520)
        self.assertFalse(mask[250,500])

    def test_pill_mask_includes_outline_and_tip_beyond_text(self):
        im=Image.new('RGB',(400,300),'white');d=ImageDraw.Draw(im)
        d.polygon([(110,128),(240,128),(258,144),(240,160),(110,160)],fill='black',outline=(170,0,220),width=2)
        d.ellipse((92,122,126,163),fill=(170,0,220))
        rs=[region('YOU CAN READ THE CHAPTER ON',135,133,90,10),region('EN-THUNDERSCANS.COM',135,145,90,10)]
        mask,box=pill_mask(np.array(im),rs)
        self.assertGreaterEqual(box[2],258)
        self.assertTrue(mask[144,257])
