import argparse
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

from review_render import archive, page_file, preview, render


def _args(root: Path, box: dict, translation: str, font_size: int, **overrides) -> argparse.Namespace:
    values = dict(
        cleaned=root / "cleaned.png", current=root / "current.png", output=root / "current.png",
        bbox=json.dumps(box), original_bbox=json.dumps(box), translation=translation,
        font="assets/fonts/wildwords/CC Wild Words Roman.ttf",
        font_size=font_size, line_spacing=0.12, alignment="center",
        color="[15,15,15]", stroke_width=0, stroke_color="[15,15,15]",
    )
    values.update(overrides)
    return argparse.Namespace(**values)


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

    def test_render_shrinks_the_font_instead_of_refusing(self):
        """A box too small for the chosen size must auto-shrink, not fail.

        Regression: the editor called `fit_text` with minimum == maximum, so a
        long translation in a box the operator had narrowed was rejected with
        "Translation does not fit with the selected font size" and the page
        never re-rendered (region stayed status EDITED).
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (400, 400), "white").save(root / "cleaned.png")
            Image.new("RGB", (400, 400), "white").save(root / "current.png")
            box = {"x": 40, "y": 40, "width": 200, "height": 80}
            # 32px cannot fit this sentence in 200x80; the renderer must fall back
            # to a smaller face (measured: fits at 18px) instead of refusing.
            render(_args(root, box, "ibu sudah bilang kalau hari ini ada murid pindahan baru kan", 32))
            with Image.open(root / "current.png") as result:
                rendered = result.crop((40, 40, 240, 120))
                self.assertLess(min(rendered.convert("L").getdata()), 80)

    def test_render_still_refuses_an_impossible_box_with_a_clear_reason(self):
        """Shrinking is bounded: a box that cannot hold the text at the floor
        must still fail, and the message must name the box and the floor."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (400, 400), "white").save(root / "cleaned.png")
            Image.new("RGB", (400, 400), "white").save(root / "current.png")
            box = {"x": 40, "y": 40, "width": 160, "height": 60}
            with self.assertRaises(ValueError) as raised:
                render(_args(root, box, "ibu sudah bilang kalau hari ini ada murid pindahan baru kan", 32))
            message = str(raised.exception)
            self.assertIn("160x60", message)
            self.assertIn("Perbesar kotak", message)

    def test_page_file_downloads_a_single_webp(self):
        """A short page downloads as one .webp named after the page."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (120, 400), "white").save(root / "001_001.png")
            written = page_file(root / "001_001.png", root / "downloads")
            self.assertEqual(written.name, "001_001.webp")
            self.assertTrue(written.exists())
            with Image.open(written) as image:
                self.assertEqual(image.format, "WEBP")
                self.assertEqual(image.size, (120, 400))

    def test_page_file_splits_a_tall_page_into_a_zip(self):
        """A page past the archive split must match the chapter ZIP members.

        The chapter archive writes `NNN_part01.webp` for a page taller than
        MAX_CONVERT_HEIGHT; the per-page download must produce the same members
        so an operator can swap one page without re-downloading the chapter.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (60, 20000), "white").save(root / "002_002.png")
            written = page_file(root / "002_002.png", root / "downloads")
            self.assertEqual(written.suffix, ".zip")
            with zipfile.ZipFile(written) as archive_file:
                names = archive_file.namelist()
            self.assertEqual(names, ["002_002_part01.webp", "002_002_part02.webp"])
            self.assertEqual(written.name, "002_002.zip")

    def test_crop_trims_a_leftover_banner_from_the_delivered_page(self):
        """A crop must shorten the delivered page, leaving the master intact.

        The complaint: source pages carry a leftover logo/banner strip at the top
        or bottom, and the operator wants it gone from the downloaded ZIP without
        editing the review master. The page download therefore applies the stored
        crop while the PNG on disk keeps its full height.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            master = root / "001_001.png"
            Image.new("RGB", (200, 1000), "white").save(master)
            written = page_file(master, root / "downloads", crop_top=200, crop_bottom=100)
            with Image.open(written) as image:
                self.assertEqual(image.size, (200, 700))
            with Image.open(master) as untouched:
                self.assertEqual(untouched.size, (200, 1000))

    def test_archive_applies_per_page_crops(self):
        """The chapter ZIP must honour each page's stored crop, not just one.

        `archive(..., database)` reads crop_top/crop_bottom per page, so a crop
        set in the editor reaches the delivered chapter archive.
        """
        import sqlite3

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("001_001.png", "002_002.png"):
                Image.new("RGB", (200, 1000), "white").save(root / name)
            database_path = root / "translator.sqlite3"
            connection = sqlite3.connect(database_path)
            connection.execute("CREATE TABLE images (output_path TEXT, crop_top INTEGER, crop_bottom INTEGER)")
            connection.execute("INSERT INTO images VALUES (?,?,?)", (str(root / "002_002.png"), 300, 0))
            connection.commit()
            connection.close()

            destination = root / "chapter.zip"
            archive(root, destination, database_path)
            with zipfile.ZipFile(destination) as output:
                self.assertEqual(output.namelist(), ["001_001.webp", "002_002.webp"])
                with Image.open(output.open("001_001.webp")) as cropped_page:
                    self.assertEqual(cropped_page.size, (200, 1000))
                with Image.open(output.open("002_002.webp")) as cropped_page:
                    self.assertEqual(cropped_page.size, (200, 700))

    def test_crop_never_delivers_an_empty_page(self):
        """An oversized crop must clamp to a sliver, not pad or empty the page.

        PIL's `crop` pads a box outside the image, which would silently grow the
        page instead of trimming it; the crop helper clamps so the result always
        has at least one row.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new("RGB", (200, 500), "white").save(root / "001_001.png")
            written = page_file(root / "001_001.png", root / "downloads", crop_top=900, crop_bottom=900)
            with Image.open(written) as image:
                self.assertEqual(image.size, (200, 1))


if __name__ == "__main__":
    unittest.main()
