#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--box", required=True, help="left,top,right,bottom")
args = parser.parse_args()
box = tuple(int(value) for value in args.box.split(","))
with Image.open(args.source) as image:
    image.crop(box).save(args.output)
