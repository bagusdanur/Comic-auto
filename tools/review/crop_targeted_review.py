from PIL import Image

p1 = Image.open("/tmp/ch29-font-test2/001.png")
p1.crop((150, 2340, 750, 2730)).save("/tmp/target-box-1.png")
p1.crop((20, 3480, 590, 3820)).save("/tmp/target-box-2.png")
p1.crop((80, 7350, 680, 7680)).save("/tmp/target-bubble.png")
p7 = Image.open("/tmp/ch29-font-test2/007.png")
p7.crop((150, 2820, 540, 3150)).save("/tmp/target-dedup.png")

