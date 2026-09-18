from PIL import Image

for source, target in [
    ("/tmp/ch29-font-final/014.png", "/tmp/footer-before.png"),
    ("/tmp/ch29-brand-final/014.png", "/tmp/footer-after.png"),
]:
    image = Image.open(source)
    image.crop((0, max(0, image.height - 1800), image.width, image.height)).save(target)

