from PIL import Image

image = Image.open("/tmp/ch29-v2/004.png")
image.crop((100, 1670, 650, 1995)).save("/tmp/ch29-v2-problem-fixed.png")
Image.open("/tmp/ch29-v2/005.png").crop((30, 20, 560, 330)).save("/tmp/ch29-v2-unsafe-005.png")
Image.open("/tmp/ch29-v2/011.png").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v2-unsafe-011.png")
Image.open("/home/ryukomik/ryukomik-auto-comic-translator/data/jobs/library/you-like-someone-with-that-face/chapter-29/2a6187df-7202-457e-88f8-e42230065d8a/original/011_ch-you-like-someone-with-that-fac_011.webp").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v2-original-011.png")
Image.open("/tmp/ch29-v2/page-011/cleaned.png").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v2-cleaned-011.png")
Image.open("/tmp/ch29-v2/page-011/mask.png").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v2-mask-011.png")
Image.open("/tmp/ch29-v2/011-v3.png").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v3-unsafe-011.png")
Image.open("/tmp/ch29-v2/011-v4.png").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v4-unsafe-011.png")
Image.open("/tmp/ch29-v2/011-v5.png").crop((80, 5850, 690, 6280)).save("/tmp/ch29-v5-unsafe-011.png")
Image.open("/tmp/ch29-v2/014.png").crop((25, 3200, 520, 3630)).save("/tmp/ch29-v2-unsafe-014.png")

