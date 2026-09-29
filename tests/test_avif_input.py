"""Halaman AVIF harus bisa masuk pipeline, bukan ditolak.

Insiden: mengimpor RAW dari Mist lewat halaman OCR selalu gagal 400
("No supported images were found"). Mist menyajikan halaman sebagai AVIF
(cdn.meowing.org/uploads/<hash>.avif), sementara `job_queue.SUPPORTED` dan
`translate.SUPPORTED_INPUTS` hanya menerima PNG/JPEG/WebP. Akibatnya seluruh
anggota ZIP dilewati, `stored` kosong, dan enqueue melempar ValueError.

Pillow 12 di venv ini mendekode AVIF secara native, jadi AVIF harus diterima
sebagai INPUT. Hasil akhir selalu ditulis ulang sebagai PNG/JPEG/WebP, jadi
AVIF tidak boleh diterima sebagai OUTPUT.
"""
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from job_queue import SUPPORTED_INPUTS, SUPPORTED_OUTPUTS, collect_inputs
import translate


def _avif_bytes(size=(24, 40), color="white") -> bytes:
    """Buat satu halaman AVIF di memori (Pillow native AVIF)."""
    import io

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="AVIF")
    return buffer.getvalue()


class AvifInputSupportTests(unittest.TestCase):
    def test_avif_diterima_sebagai_input(self):
        self.assertIn(".avif", SUPPORTED_INPUTS)
        self.assertIn(".avif", translate.SUPPORTED_INPUTS)

    def test_avif_tidak_diterima_sebagai_output(self):
        # AVIF hanya untuk dibaca; halaman selalu dinormalkan ke PNG.
        self.assertNotIn(".avif", SUPPORTED_OUTPUTS)
        self.assertNotIn(".avif", translate.SUPPORTED_OUTPUTS)

    def test_zip_avif_tidak_lagi_kosong(self):
        """Regresi: ZIP berisi AVIF dulu menghasilkan ValueError."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mist-chapter.zip"
            with zipfile.ZipFile(source, "w") as archive:
                for index in range(1, 4):
                    archive.writestr(f"{index:03d}.avif", _avif_bytes())
            destination = root / "stored"
            destination.mkdir()
            stored = collect_inputs([source], destination)
            self.assertEqual(len(stored), 3)
            self.assertTrue(all(item.suffix == ".avif" for item in stored))

    def test_halaman_avif_bisa_dibaca_pipeline(self):
        """Setiap file yang lolos collect_inputs harus bisa dibuka PIL."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chapter.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("001.avif", _avif_bytes())
            destination = root / "stored"
            destination.mkdir()
            stored = collect_inputs([source], destination)
            with Image.open(stored[0]) as image:
                image.verify()
            with Image.open(stored[0]) as image:
                self.assertEqual(image.size, (24, 40))
                self.assertEqual(image.convert("RGB").mode, "RGB")

    def test_campuran_avif_dan_png_tetap_masuk(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            png = root / "001.png"
            Image.new("RGB", (16, 16), "white").save(png)
            source = root / "mixed.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("001.avif", _avif_bytes())
                archive.write(png, "002.png")
            destination = root / "stored"
            destination.mkdir()
            stored = collect_inputs([source], destination)
            self.assertEqual(len(stored), 2)
            self.assertEqual({item.suffix for item in stored}, {".avif", ".png"})


if __name__ == "__main__":
    unittest.main()
