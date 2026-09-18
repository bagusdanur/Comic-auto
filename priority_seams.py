"""Lossless RAW boundary relocation for the priority Heavenly Demon title."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

PRIORITY_SLUG='who-s-threatening-you-with-a-knife-to-make-you-a-heavenly-demon'

def boundary_offset(left,right,limit=1200):
    if left.shape[1]!=right.shape[1]:return 0
    before=min(limit,left.shape[0]-100);after=min(limit,right.shape[0]-100)
    if min(before,after)<16:return 0
    strip=np.concatenate((left[-before:],right[:after]),axis=0)
    white=(strip.min(axis=2)>240).mean(axis=1)>.99
    safe=np.convolve(white.astype(int),np.ones(16,dtype=int),mode='valid')==16
    candidates=np.where(safe)[0]+8-before
    return int(min(candidates,key=abs)) if len(candidates) else 0

def prepare_inputs(originals,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    signature=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in originals]
    manifest=directory/'manifest.json'
    if manifest.exists():
        cached=json.loads(manifest.read_text())
        if cached['signature']==[list(v) for v in signature] and all(Path(p).exists() for p in cached['paths']):return [Path(p) for p in cached['paths']]
    sizes=[]
    for p in originals:
        with Image.open(p) as im:sizes.append(im.size)
    shifts=[0]
    for i in range(len(originals)-1):
        with Image.open(originals[i]) as a,Image.open(originals[i+1]) as b:
            shifts.append(boundary_offset(np.array(a.convert('RGB')),np.array(b.convert('RGB'))))
    shifts.append(0);outputs=[]
    for i,p in enumerate(originals):
        start,end=shifts[i],sizes[i][1]+shifts[i+1]
        if start==0 and end==sizes[i][1]:outputs.append(p);continue
        assert start<end
        parts=[]
        if start<0:
            with Image.open(originals[i-1]) as im:parts.append(im.convert('RGB').crop((0,im.height+start,im.width,im.height)))
        with Image.open(p) as im:parts.append(im.convert('RGB').crop((0,max(0,start),im.width,min(im.height,end))))
        if end>sizes[i][1]:
            with Image.open(originals[i+1]) as im:parts.append(im.convert('RGB').crop((0,0,im.width,end-sizes[i][1])))
        merged=Image.new('RGB',(sizes[i][0],sum(im.height for im in parts)));y=0
        for part in parts:merged.paste(part,(0,y));y+=part.height
        target=directory/(p.stem+'.png');merged.save(target);outputs.append(target)
    manifest.write_text(json.dumps(dict(signature=signature,paths=list(map(str,outputs)),boundary_shifts=shifts,source_sizes=sizes),indent=2))
    return outputs
