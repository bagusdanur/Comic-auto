#!/usr/bin/env python3
"""Print the detected stacked source-promo boundary for one RAW/OCR pair."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from branding_v52 import stacked_source_promo_header

parser=argparse.ArgumentParser()
parser.add_argument("image",type=Path)
parser.add_argument("ocr",type=Path)
args=parser.parse_args()
with Image.open(args.image) as opened:image=np.asarray(opened.convert("RGB"))
regions=json.loads(args.ocr.read_text(encoding="utf-8")).get("regions",[])
print(stacked_source_promo_header(image,regions))
