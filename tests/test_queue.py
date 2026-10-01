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

    def add_job(self, job_id: str, created: float = 1.0, position: float | None = None):
        self.db.execute(
            "INSERT INTO jobs(id,status,input_path,output_path,options_json,created_at,updated_at,queue_position) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (job_id, "QUEUED", "/input.png", "/output.png", json.dumps({}), created, time.time(), position),
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

    def test_claim_follows_created_at_when_no_manual_order(self):
        # Backwards compatibility: rows without a queue_position keep plain FIFO,
        # which is what every job enqueued before this column existed relies on.
        self.add_job("later", created=200)
        self.add_job("earlier", created=100)
        claimed = claim_next(self.db, 3000, 2000)
        assert claimed is not None
        self.assertEqual(claimed["id"], "earlier")

    def test_queue_position_overrides_created_at(self):
        # The whole point of the feature: a job enqueued last can run first.
        self.add_job("first-uploaded", created=100, position=100)
        self.add_job("promoted", created=200, position=1)
        claimed = claim_next(self.db, 3000, 2000)
        assert claimed is not None
        self.assertEqual(claimed["id"], "promoted")

    def test_connect_backfills_positions_for_existing_rows(self):
        # Simulate a database created before queue_position existed: connect()
        # must add the column and give every row its created_at as the position.
        self.add_job("old", created=42, position=None)
        self.db.execute("UPDATE jobs SET queue_position=NULL WHERE id='old'")
        self.db.commit()
        reopened = connect(self.path)
        try:
            position = reopened.execute("SELECT queue_position FROM jobs WHERE id='old'").fetchone()[0]
            self.assertEqual(position, 42)
        finally:
            reopened.close()

    def test_enqueue_sets_position_so_new_jobs_queue_behind(self):
        # A freshly enqueued job must not jump ahead of an older one: created_at
        # and queue_position are both "now", so plain order is preserved.
        self.add_job("waiting", created=100, position=100)
        self.add_job("fresh", created=500, position=500)
        claimed = claim_next(self.db, 3000, 2000)
        assert claimed is not None
        self.assertEqual(claimed["id"], "waiting")


if __name__ == "__main__":
    unittest.main()
