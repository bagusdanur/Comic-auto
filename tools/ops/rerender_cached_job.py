#!/usr/bin/env python3
"""Safely rerender a completed V5 job from cached OCR/translation/removal data."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from brand_postprocess import apply_branding
from webp_archive import archive_webp


def run(command):
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('job_root',type=Path)
    args=parser.parse_args();job=args.job_root.resolve()
    originals=sorted(p for p in (job/'original').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
    outputs=[]
    for index,source in enumerate(originals,1):
        page=job/'metadata'/f'page-{index:03d}'
        temporary=page/'typeset-style-v5.tmp.png';report=page/'typeset-style-v5.tmp.json';quality=page/'quality-style-v5.tmp.json'
        command=[sys.executable,str(ROOT/'typeset_test.py'),str(page/'cleaned.png'),str(page/'translations.json'),
            '--ocr-json',str(page/'ocr-result.json'),'--reference-image',str(source),'--removal-report',str(page/'removal-report.json'),
            '--output',str(temporary),'--report',str(report),'--min-font-size','22','--min-scale-ratio','0.52',
            '--layout-engine','v5','--surface-map',str(page/'surface-map.json'),'--hard-clip']
        run(command)
        run([sys.executable,str(ROOT/'quality_pipeline_v2.py'),'validate',str(source),str(page/'cleaned.png'),str(temporary),
             str(page/'mask-debug.png'),str(page/'surface-map.json'),'--report',str(quality)])
        verdict=json.loads(quality.read_text())
        if verdict.get('status')!='PASS':raise RuntimeError(f'{page.name} style render failed QC: {verdict.get("review_reasons")}')
        os.replace(temporary,page/'typeset.png');os.replace(report,page/'typeset-report.json');os.replace(quality,page/'quality-report.json')
        destination=job/'output'/f'{source.stem}.png';shutil.copy2(page/'typeset.png',destination)
        branding=apply_branding(destination,page/'ocr-result.json',ROOT/'assets/branding/watermark.png',ROOT/'assets/branding/watermark-text.png',
                                first_page=index==1,banner_path=ROOT/'assets/branding/banner.png' if index==len(originals) else None)
        (page/'branding-report.json').write_text(json.dumps(branding,ensure_ascii=False,indent=2),encoding='utf-8')
        outputs.append(destination);print(f'{page.name} PASS',flush=True)
    count=archive_webp(outputs,job/'output/chapter-translated.zip')
    print(json.dumps({'status':'COMPLETED','pages':len(outputs),'archive_entries':count}))


if __name__=='__main__':main()
