import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from job_queue import collect_inputs
from queue_db import connect


class BatchQueueTests(unittest.TestCase):
    def test_zip_images_are_validated_and_sorted_into_safe_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "chapter.zip"
            page = root / "page.png"
            Image.new("RGB", (32, 48), "white").save(page)
            with zipfile.ZipFile(source, "w") as archive:
                archive.write(page, "chapter/002 page.png")
                archive.write(page, "chapter/001.png")
                archive.writestr("notes.txt", "ignored")
            destination = root / "stored"
            destination.mkdir()
            stored = collect_inputs([source], destination)
            self.assertEqual(len(stored), 2)
            self.assertTrue(all(item.parent == destination for item in stored))
            self.assertTrue(all(item.name[:3].isdigit() for item in stored))

    def test_zip_slip_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "unsafe.zip"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("../escape.png", b"not-an-image")
            destination = root / "stored"
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, "Unsafe ZIP path"):
                collect_inputs([source], destination)

    def test_schema_has_per_image_metadata_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = connect(Path(temporary) / "queue.sqlite3")
            try:
                columns = {row[1] for row in database.execute("PRAGMA table_info(images)")}
                self.assertIn("metadata_path", columns)
            finally:
                database.close()


if __name__ == "__main__":
    unittest.main()
