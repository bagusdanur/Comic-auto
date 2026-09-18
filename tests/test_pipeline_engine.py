import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline_engine import require_implemented, requested_engine
from tools.audit.build_pipeline_baseline import build_manifest


class PipelineEngineTests(unittest.TestCase):
    def test_default_is_current_legacy_pipeline(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(requested_engine(), "legacy")

    def test_unknown_engine_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "Unknown pipeline engine"):
            requested_engine("typo")

    def test_v3_engine_is_available_after_shadow_audit(self):
        self.assertEqual(require_implemented("v3"), "v3")

    def test_v5_engine_is_available(self):
        self.assertEqual(require_implemented("v5"), "v5")

    def test_shadow_engine_cannot_run_directly(self):
        with self.assertRaisesRegex(RuntimeError, "not implemented"):
            require_implemented("shadow")

    def test_baseline_references_files_without_copying_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            page = root / "metadata" / "page-001"
            page.mkdir(parents=True)
            (page / "typeset.png").write_bytes(b"image")
            (page / "pipeline-status.json").write_text(
                '{"status":"COMPLETED","review_ids":[],"durations":{"OCR":1.2}}',
                encoding="utf-8",
            )
            manifest = build_manifest(root)
            self.assertEqual(manifest["page_count"], 1)
            self.assertEqual(manifest["pages"][0]["pipeline_engine"], "legacy-unversioned")
            self.assertEqual(manifest["pages"][0]["files"][0]["bytes"], 5)
            self.assertNotIn("sha256", manifest["pages"][0]["files"][0])


if __name__ == "__main__":
    unittest.main()
