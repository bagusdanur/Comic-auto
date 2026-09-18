import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

root = Path("/tmp/memory-smoke")
root.mkdir(exist_ok=True)
font = ImageFont.truetype("assets/fonts/animeace/animeace2bb_tt/animeace2_reg.ttf", 38)
for name, text in (("raw", "USE SHADOW STEP!"), ("translated", "PAKAI SHADOW STEP!")):
    image = Image.new("RGB", (700, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((90, 180, 610, 430), radius=70, outline="black", width=5)
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((700 - (box[2]-box[0]))/2, 280), text, font=font, fill="black")
    page = root / f"{name}.png"
    image.save(page)
    with zipfile.ZipFile(root / f"{name}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(page, "001.png")

