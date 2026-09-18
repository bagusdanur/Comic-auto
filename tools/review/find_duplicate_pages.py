#!/usr/bin/env python3
"""Report exact and strongly similar image pairs in a directory."""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
files = sorted(p for p in args.directory.iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'})
rows=[]
for path in files:
    image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    thumb=cv2.resize(image,(64,256),interpolation=cv2.INTER_AREA)
    rows.append((path,hashlib.sha256(path.read_bytes()).hexdigest(),thumb))
for i,(left,left_hash,left_thumb) in enumerate(rows):
    for right,right_hash,right_thumb in rows[i+1:]:
        mae=float(np.mean(np.abs(left_thumb.astype(np.int16)-right_thumb.astype(np.int16))))
        if left_hash==right_hash or mae<18:
            print(f"{left.name}\t{right.name}\texact={left_hash==right_hash}\tmae={mae:.2f}")
print(f"files={len(files)}")
