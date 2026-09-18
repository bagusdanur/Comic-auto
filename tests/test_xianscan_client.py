import unittest

import cv2
import numpy as np
import base64

from xianscan_client import (
    XianScanClient, allowed_mask, match_regions, normalize_analyze_response,
    payload_for_regions,
)


class _Response:
    def __init__(self, content=b"", payload=None):
        self.content = content
        self._payload = payload or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, analyze_payload=None):
        self.analyze_payload = analyze_payload
        self.urls = []

    def get(self, *_args, **_kwargs):
        return _Response(payload={"status": "ok"})

    def post(self, url, **kwargs):
        self.urls.append(url)
        if url.endswith("/pages/analyze") or url.endswith("/pages/detect"):
            return _Response(payload=self.analyze_payload)
        raw = np.frombuffer(kwargs["files"]["image"][1], dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        ok, encoded = cv2.imencode(".png", image)
        assert ok
        return _Response(encoded.tobytes())


class XianScanClientTests(unittest.TestCase):
    def test_masked_inpaint_validates_exact_pixel_boundary(self):
        image = np.full((30, 40, 3), 127, dtype=np.uint8)
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[5:10, 6:12] = 255
        result = XianScanClient("http://local", session=_Session()).inpaint_mask(image, mask)
        self.assertTrue(np.array_equal(result, image))

    def test_masked_inpaint_splits_oversized_masks_for_every_caller(self):
        image = np.full((100, 40, 3), 127, dtype=np.uint8)
        mask = np.full((100, 40), 255, dtype=np.uint8)
        session = _Session()
        result = XianScanClient("http://local", session=session).inpaint_mask(image, mask)
        calls = [url for url in session.urls if url.endswith("/pages/inpaint-mask")]
        self.assertEqual(len(calls), 7)
        self.assertTrue(np.array_equal(result, image))

    def test_tiled_analysis_offsets_and_deduplicates_overlap(self):
        class TileClient(XianScanClient):
            def __init__(self):
                self.calls = 0

            def analyze(self, tile):
                local_y = 220 if self.calls == 0 else 20
                self.calls += 1
                return {
                    "width": tile.shape[1], "height": tile.shape[0], "backend": "comic-ctd",
                    "region_count": 1,
                    "regions": [{
                        "id": "same", "text": "HELLO", "confidence": 0.9,
                        "bbox": {"x": 3, "y": local_y, "width": 10, "height": 10},
                        "polygon": [[3, local_y], [13, local_y],
                                    [13, local_y + 10], [3, local_y + 10]],
                    }],
                }

        image = np.zeros((600, 40, 3), dtype=np.uint8)
        result = TileClient().analyze_tiled(image, tile_height=400, overlap=200)
        self.assertEqual(result["tile_count"], 2)
        self.assertEqual([item["bbox"]["y"] for item in result["regions"]], [220])

    def test_analyze_normalizes_sidecar_regions(self):
        payload = {
            "width": 40, "height": 30, "backend": "comic-ctd",
            "regions": [{
                "id": "r7", "box": {"x": 4, "y": 5, "w": 15, "h": 10},
                "polygon": [[4, 5], [19, 5], [19, 15], [4, 15]],
                "text": " hello ", "confidence": 0.91, "angle": 2.5,
            }],
        }
        image = np.zeros((30, 40, 3), dtype=np.uint8)
        session = _Session(analyze_payload=payload)
        result = XianScanClient("http://local", session=session).analyze(image)
        self.assertEqual(result["region_count"], 1)
        self.assertEqual(result["regions"][0]["id"], "xianscan:r7")
        self.assertEqual(result["regions"][0]["text"], "hello")
        self.assertEqual(session.urls[0], "http://local/pages/analyze")

    def test_detect_accepts_empty_text_and_decodes_mask(self):
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[5:10, 6:12] = 255
        ok, encoded = cv2.imencode(".png", mask)
        self.assertTrue(ok)
        payload = {
            "width": 40, "height": 30, "backend": "comic-ctd",
            "regions": [{
                "id": "r1", "box": {"x": 4, "y": 5, "w": 15, "h": 10},
                "polygon": [], "text": "", "confidence": 0.88,
            }],
            "mask_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
        }
        image = np.zeros((30, 40, 3), dtype=np.uint8)
        result = XianScanClient("http://local", session=_Session(payload)).detect(
            image, include_mask=True
        )
        self.assertEqual(result["region_count"], 1)
        self.assertEqual(result["regions"][0]["text"], "")
        self.assertEqual(result["text_mask"].shape, (30, 40))

    def test_normalizer_rejects_dimension_changes(self):
        with self.assertRaisesRegex(ValueError, "changed dimensions"):
            normalize_analyze_response({"width": 9, "height": 8}, (10, 10))

    def test_region_matching_is_one_to_one(self):
        primary = [
            {"id": "p1", "bbox": {"x": 0, "y": 0, "width": 10, "height": 10}},
            {"id": "p2", "bbox": {"x": 1, "y": 1, "width": 10, "height": 10}},
        ]
        secondary = [{"id": "s1", "bbox": {"x": 0, "y": 0, "width": 10, "height": 10}}]
        matches = match_regions(primary, secondary)
        self.assertEqual(matches[0]["secondary_id"], "s1")
        self.assertIsNone(matches[1]["secondary_id"])

    def test_payload_and_allowed_mask_follow_source_polygon(self):
        regions = [{
            "id": "r1", "bbox": {"x": 10, "y": 20, "width": 30, "height": 15},
            "polygon": [[10, 20], [40, 20], [40, 35], [10, 35]],
        }]
        payload = payload_for_regions(regions)
        self.assertEqual(payload[0]["box"], {"x": 10, "y": 20, "w": 30, "h": 15})
        mask = allowed_mask((80, 80), regions, dilation=0)
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(mask[5, 5], 0)

    def test_clean_validates_roundtrip_dimensions(self):
        image = np.full((64, 96, 3), 127, dtype=np.uint8)
        client = XianScanClient("http://local", session=_Session())
        result = client.clean(image, [])
        self.assertTrue(np.array_equal(result, image))


if __name__ == "__main__":
    unittest.main()
