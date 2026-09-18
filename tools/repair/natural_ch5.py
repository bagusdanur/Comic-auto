"""Audited Chapter 5 wording corrections and standalone-promo replacement."""
import json,shutil,sqlite3,subprocess,sys,zipfile
from pathlib import Path
from PIL import Image
from brand_postprocess import apply_branding
from branding_coordinates import project_region
from webp_archive import archive_webp

changes={
 2:{3:'Jangan mengeluh! Masa seorang Sword Master tidak bisa menghindar?',4:'Jangan menghalangi! Jaga saja barang bawaannya!',8:'Ugh... Bajingan-bajingan ini berisik banget.'},
 3:{4:'Aku Erica, tentara bayaran kelas S dari Guild Pechenik.',8:'Tidak nyangka dapat rezeki begini di tempat terpencil!'},
 5:{5:'Selama ini aku... benar-benar tidak paham.',9:'Biar aku yang mengurus akibatnya,',10:'kamu bunuh saja mereka.'},
 7:{5:'KALAU INI MISI RESMI, BAYARANNYA SETARA MISI KELAS A!',6:'DUA TENTARA BAYARAN KELAS S MALAH KERJA GRATIS!',8:'KAMI TIDAK TAHU HARUS MEMBALAS KEBAIKAN KALIAN DENGAN APA...',10:'Tidak usah repot-repot!',11:'Tapi... kalau memang ingin membalasnya, mampir saja ke Guild Pechenik.',13:'PAK! ROMBONGAN PEDAGANG SUDAH SIAP BERANGKAT!'},
 9:{1:'Teman dari serikat pedagang, tapi...'}
}
jid='44e06345-7314-48b4-b369-8a300f21b23d'
c=sqlite3.connect('data/translator.sqlite3');c.row_factory=sqlite3.Row
job=c.execute('select * from jobs where id=?',(jid,)).fetchone();assert job['status']=='COMPLETED'
rows=c.execute('select * from images where job_id=? order by filename',(jid,)).fetchall();assert len(rows)==10
work=Path('artifacts/ch5-natural-fix');work.mkdir(exist_ok=False)
(work/'db-before.json').write_text(json.dumps({'images':[dict(r) for r in rows],'regions':[dict(r) for r in c.execute('select r.* from regions r join images i on r.image_id=i.id where i.job_id=?',(jid,))]},indent=2))
staged={}
for number in [*changes,10]:
 row=rows[number-1];page=Path(row['metadata_path']);stage=work/f'page-{number:03d}'
 shutil.copytree(page,stage)
 shutil.copy2(row['output_path'],stage/'before-output.png')
 data=json.loads((stage/'translations.json').read_text())
 for g in data['groups']:
  n=int(g['id'].split('-')[-1])
  if n in changes.get(number,{}):g['translation']=changes[number][n]
 data['wording_revision']='natural-id-ch5-v1'
 (stage/'translations.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
 if number!=10:
  raw=json.loads((stage/'pipeline-status.json').read_text())['input']
  subprocess.run([sys.executable,'typeset_test.py',str(stage/'cleaned.png'),str(stage/'translations.json'),'--ocr-json',str(stage/'ocr-result.json'),'--reference-image',raw,'--removal-report',str(stage/'removal-report.json'),'--surface-map',str(stage/'surface-map.json'),'--hard-clip','--layout-engine','v5','--output',str(stage/'typeset.png'),'--report',str(stage/'typeset-report.json')],check=True,stdout=subprocess.DEVNULL)
 out=stage/'final.png';shutil.copy2(stage/'typeset.png',out)
 report=apply_branding(out,stage/'ocr-result.json',Path('assets/branding/watermark.png'),Path('assets/branding/watermark-text.png'),banner_path=Path('assets/branding/banner.png') if number==10 else None)
 if number==10:assert report.get('standalone_promo_removed')
 (stage/'branding-report.json').write_text(json.dumps(report,indent=2))
 staged[number]=(stage,out,report,data)
 print('STAGED',number,flush=True)
# Create the complete replacement ZIP before publishing any edited page.
zip_inputs=work/'zip-pages';zip_inputs.mkdir()
for number,row in enumerate(rows,1):shutil.copy2(staged[number][1] if number in staged else row['output_path'],zip_inputs/Path(row['output_path']).name)
archive_webp(list(zip_inputs.glob('*.png')),work/'chapter.zip')
with zipfile.ZipFile(work/'chapter.zip') as z:assert len(z.namelist())==10 and z.testzip() is None
shutil.copy2(job['output_path'],work/'before-chapter.zip')
for number,(stage,out,report,data) in staged.items():
 row=rows[number-1];page=Path(row['metadata_path'])
 for name in ['translations.json','typeset.png','typeset-report.json','branding-report.json']:
  shutil.copy2(page/name,stage/('before-'+name));shutil.copy2(stage/name,page/name)
 shutil.copy2(out,row['output_path'])
 results={r['id']:r for r in json.loads((stage/'typeset-report.json').read_text())['results']}
 for g in data['groups']:
  projected=project_region(results[g['id']],g['bbox'],report)
  if projected is None:
   c.execute('delete from regions where image_id=? and source_text=?',(row['id'],g['text']))
  else:
   box,item=projected
   c.execute('update regions set translation=?,bbox_json=?,metadata_json=? where image_id=? and source_text=?',(g['translation'],json.dumps(box),json.dumps(item),row['id'],g['text']))
 with Image.open(out) as im:c.execute('update images set width=?,height=?,region_count=(select count(*) from regions where image_id=?) where id=?',(*im.size,row['id'],row['id']))
 preview=Path(row['output_path']).parent/'previews'/f"{row['id']}.jpg"
 if preview.exists():shutil.move(str(preview),str(stage/'before-preview.jpg'))
shutil.copy2(work/'chapter.zip',job['output_path']);c.commit()
print('COMPLETE: 10 WebP quality95 pages, 18 audited wording edits, standalone promo replaced; backups:',work,flush=True)
