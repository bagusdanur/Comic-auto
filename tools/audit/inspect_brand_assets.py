from PIL import Image
from pathlib import Path

for path in Path("assets/branding").glob("*.png"):
    with Image.open(path) as image:
        alpha = image.getextrema()[-1] if image.mode == "RGBA" else "no-alpha"
        print(path.name, image.mode, image.size, alpha)

