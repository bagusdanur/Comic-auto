from PIL import Image

for source, target in [
    ("assets/branding/banner.png", "/tmp/banner-preview.png"),
    ("assets/branding/watermark.png", "/tmp/logo-preview.png"),
]:
    image = Image.open(source).convert("RGBA")
    width = 760
    image = image.resize((width, round(image.height * width / image.width)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", image.size, "white")
    canvas.alpha_composite(image)
    canvas.convert("RGB").save(target)

final = Image.open("/tmp/ch29-brand-final/014.png")
final.crop((0, final.height - 417, final.width, final.height)).save("/tmp/exact-banner.png")

