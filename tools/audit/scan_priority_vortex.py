import sqlite3,pathlib,json,cv2,numpy as np
from PIL import Image
root=pathlib.Path(sqlite3.connect('data/translator.sqlite3').execute("select input_path from jobs where id='adbfe8cc-304c-4fdc-8287-40e6ac878345'").fetchone()[0]).parent.parent
paths=sorted((root/'original').iterdir())
def red(a):
 h=cv2.cvtColor(a,cv2.COLOR_RGB2HSV);return (((h[:,:,0]<12)|(h[:,:,0]>170))&(h[:,:,1]>85)&(h[:,:,2]>45)).astype(np.uint8)*255
template=red(np.array(Image.open(paths[1]).convert('RGB'))[0:154,18:161])
for index,p in enumerate(paths):
 a=np.array(Image.open(p).convert('RGB'));binary=red(a);candidates=[]
 for size in range(40,117,4):
  t=cv2.resize(template,(size,round(size*template.shape[0]/template.shape[1])))
  score=cv2.matchTemplate(binary,t,cv2.TM_CCOEFF_NORMED)
  ys,xs=np.where(score>.63)
  candidates.extend((float(score[y,x]),int(x),int(y),size,t.shape[0]) for y,x in zip(ys,xs))
 chosen=[]
 for v in sorted(candidates,reverse=True):
  if any(abs(v[1]-z[1])<70 and abs(v[2]-z[2])<70 for z in chosen):continue
  chosen.append(v)
 print(index+1,chosen,flush=True)
 if index==5:
  whites=(a[:2200].min(axis=2)>240).mean(axis=1)
  print('seam white rows',[(i,float(whites[i])) for i in range(0,2200,50) if whites[i]>.98],flush=True)
