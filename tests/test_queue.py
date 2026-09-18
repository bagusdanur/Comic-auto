import json
import tempfile
import time
import unittest
from pathlib import Path

from queue_db import claim_next, connect


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "queue.sqlite3"
        self.db = connect(self.path)

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def add_job(self, job_id: str, created: float = 1.0):
        self.db.execute(
            "INSERT INTO jobs(id,status,input_path,output_path,options_json,created_at,updated_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (job_id, "QUEUED", "/input.png", "/output.png", json.dumps({}), created, time.time()),
        )
        self.db.commit()

    def test_resource_wait_then_claim(self):
        self.add_job("one")
        self.assertIsNone(claim_next(self.db, 1999, 2000))
        self.assertEqual(self.db.execute("SELECT status FROM jobs WHERE id='one'").fetchone()[0], "WAITING_RESOURCE")
        claimed = claim_next(self.db, 3000, 2000)
        self.assertEqual(claimed["id"], "one")
        self.assertEqual(claimed["status"], "PROCESSING")

    def test_only_one_processing_across_connections(self):
        self.add_job("one", 1)
        self.add_job("two", 2)
        second = connect(self.path)
        try:
            self.assertEqual(claim_next(self.db, 3000, 2000)["id"], "one")
            self.assertIsNone(claim_next(second, 3000, 2000))
            self.assertEqual(second.execute("SELECT status FROM jobs WHERE id='two'").fetchone()[0], "QUEUED")
        finally:
            second.close()


if __name__ == "__main__":
    unittest.main()
