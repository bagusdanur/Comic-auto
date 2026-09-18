from PIL import Image, ImageDraw

first = Image.open("/tmp/brand-smoke/001.png")
first.crop((0, 0, first.width, 1300)).save("/tmp/brand-smoke/first-top.png")
first.crop((0, 3050, first.width, 3450)).save("/tmp/brand-smoke/first-old-mark.png")
last = Image.open("/tmp/brand-smoke/014.png")
last.crop((0, last.height - 900, last.width, last.height)).save("/tmp/brand-smoke/last-banner.png")

