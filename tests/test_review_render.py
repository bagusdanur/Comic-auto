import argparse
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from review_render import archive, preview, render


class ReviewRenderTest(unittest.TestCase):
    def test_preview_and_single_region_render_preserve_canvas(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cleaned = root / "cleaned.png"
            current = root / "current.png"
            preview_path = root / "preview.jpg"
            base = Image.new("RGB", (200, 300), "white")
            base.save(cleaned)
            before = base.copy()
            draw = ImageDraw.Draw(before)
            draw.rectangle((40, 80, 160, 150), fill="white")
            before.save(current)

            preview(current, preview_path, 100)
            with Image.open(preview_path) as thumb:
                self.assertEqual(thumb.size, (100, 150))

            box = {"x": 40, "y": 80, "width": 120, "height": 70}
            render(argparse.Namespace(
                cleaned=cleaned, current=current, output=current,
                bbox=__import__("json").dumps(box), original_bbox=__import__("json").dumps(box),
                translation="uji teks", font="assets/fonts/wildwords/CC Wild Words Roman.ttf",
                font_size=20, line_spacing=0.12, alignment="center", color="[255,255,255]",
                stroke_width=2, stroke_color="[255,255,255]",
            ))
            with Image.open(current) as result:
                self.assertEqual(result.size, (200, 300))
                outside_before = base.crop((0, 0, 200, 70))
                outside_after = result.crop((0, 0, 200, 70))
                self.assertIsNone(ImageChops.difference(outside_before, outside_after).getbbox())
                self.assertIsNotNone(ImageChops.difference(base.crop((40, 80, 160, 150)), result.crop((40, 80, 160, 150))).getbbox())
                rendered = result.crop((40, 80, 160, 150))
                darkest = min(pixel for pixel in rendered.convert("L").getdata())
                self.assertLess(darkest, 80)
            archive(root, root / "chapter.zip")
            with __import__("zipfile").ZipFile(root / "chapter.zip") as output:
                self.assertEqual(output.namelist(), ["cleaned.webp", "current.webp"])


if __name__ == "__main__":
    unittest.main()
