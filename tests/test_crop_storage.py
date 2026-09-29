"""Crop storage: the DB columns, the loader the archive uses, and migration.

A leftover source banner is removed from the delivered page by storing a crop on
the image row. Two things must hold: an old database (created before the crop
columns existed) must be migrated in place, and `load_crops` must map the stored
rows back to the rendered PNGs the archive iterates.
"""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from queue_db import connect, load_crops


class CropStorageTest(unittest.TestCase):
    def test_connect_migrates_a_legacy_images_table(self):
        """A pre-existing DB without crop columns must gain them, not crash."""
        with tempfile.TemporaryDirectory() as temporary:
            database_path = Path(temporary) / "translator.sqlite3"
            # Simulate an old database: images table lacking the crop columns.
            legacy = sqlite3.connect(database_path)
            legacy.execute(
                "CREATE TABLE images (id TEXT PRIMARY KEY, job_id TEXT, filename TEXT, "
                "width INTEGER, height INTEGER, status TEXT, output_path TEXT, "
                "metadata_path TEXT, region_count INTEGER NOT NULL DEFAULT 0)"
            )
            legacy.execute("INSERT INTO images VALUES ('a','j','001.png',10,10,'TYPESET','/x/001.png',NULL,0)")
            legacy.commit()
            legacy.close()

            database = connect(database_path)
            try:
                columns = {row[1] for row in database.execute("PRAGMA table_info(images)")}
                self.assertIn("crop_top", columns)
                self.assertIn("crop_bottom", columns)
                self.assertIn("crop_updated_at", columns)
                row = database.execute("SELECT crop_top,crop_bottom FROM images WHERE id='a'").fetchone()
                self.assertEqual((row[0], row[1]), (0, 0))
            finally:
                database.close()

    def test_load_crops_maps_output_path_to_file_name(self):
        """Only cropped pages appear, keyed by file name, matched by path."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "001_001.png"
            second = root / "002_002.png"
            for path in (first, second):
                path.write_bytes(b"")
            database_path = root / "translator.sqlite3"
            database = connect(database_path)
            try:
                database.execute(
                    "INSERT INTO jobs(id,status,input_path,output_path,options_json,stage,progress,"
                    "review_count,created_at,updated_at) VALUES('j','COMPLETED','/in','/out.zip','{}','DONE',1,0,0,0)"
                )
                for image_id, path, top, bottom in (
                    ("a", first, 0, 0),
                    ("b", second, 120, 40),
                ):
                    database.execute(
                        "INSERT INTO images(id,job_id,filename,width,height,status,output_path,region_count,"
                        "crop_top,crop_bottom) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (image_id, "j", path.name, 10, 10, "TYPESET", str(path), 0, top, bottom),
                    )
                database.commit()
                crops = load_crops(database, [first, second])
            finally:
                database.close()
            # The uncropped page is omitted so callers can treat a miss as (0, 0).
            self.assertEqual(crops, {"002_002.png": (120, 40)})

    def test_load_crops_tolerates_a_database_without_crop_columns(self):
        """A DB whose images table predates the columns yields no crops."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = root / "001_001.png"
            page.write_bytes(b"")
            database_path = root / "legacy.sqlite3"
            legacy = sqlite3.connect(database_path)
            legacy.row_factory = sqlite3.Row
            legacy.execute("CREATE TABLE images (id TEXT, output_path TEXT)")
            legacy.commit()
            self.assertEqual(load_crops(legacy, [page]), {})
            legacy.close()


if __name__ == "__main__":
    unittest.main()
