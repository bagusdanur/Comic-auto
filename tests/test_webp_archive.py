import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from unittest.mock import patch

from webp_archive import archive_webp


class WebpArchiveTests(unittest.TestCase):
    def test_default_encoder_quality_is_100(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'001.png'
            Image.new('RGB',(8,20),'white').save(source)
            with patch('webp_archive._webp_bytes',return_value=b'test') as encode:
                archive_webp([source],root/'chapter.zip')
                self.assertEqual(encode.call_args.args[1],100)

    def test_converts_and_splits_like_dashboard_converter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            short = root / "001.png"
            tall = root / "002.png"
            Image.new("RGB", (8, 20), "white").save(short)
            Image.new("RGB", (8, 16001), "black").save(tall)
            destination = root / "chapter.zip"
            self.assertEqual(archive_webp([tall, short], destination), 3)
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(archive.namelist(), ["001.webp", "002_part01.webp", "002_part02.webp"])
                for name in archive.namelist():
                    self.assertEqual(Image.open(archive.open(name)).format, "WEBP")


if __name__ == "__main__":
    unittest.main()
