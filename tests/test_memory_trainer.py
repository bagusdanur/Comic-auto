import unittest

from memory_trainer import match_groups


def page(texts):
    return {
        "width": 1000, "height": 2000,
        "regions": [
            {"id": str(index), "text": text, "confidence": .99, "bbox": box}
            for index, (text, box) in enumerate(texts)
        ],
    }


class MemoryTrainerTests(unittest.TestCase):
    def test_regions_pair_by_normalized_position(self):
        raw = page([
            ("Shadow Step!", {"x": 100, "y": 200, "width": 300, "height": 80}),
            ("Are you okay?", {"x": 600, "y": 900, "width": 250, "height": 80}),
        ])
        translated = page([
            ("SHADOW STEP!", {"x": 110, "y": 205, "width": 310, "height": 85}),
            ("KAMU NGGAK APA-APA?", {"x": 590, "y": 910, "width": 270, "height": 85}),
        ])
        pairs = match_groups(raw, translated)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0]["source"], "Shadow Step!")
        self.assertEqual(pairs[1]["translation"], "KAMU NGGAK APA-APA?")
        self.assertGreater(pairs[0]["confidence"], .8)

    def test_distant_region_is_not_forced_into_pair(self):
        raw = page([("Hello", {"x": 20, "y": 20, "width": 100, "height": 50})])
        translated = page([("HALO", {"x": 800, "y": 1800, "width": 100, "height": 50})])
        self.assertEqual(match_groups(raw, translated), [])


if __name__ == "__main__":
    unittest.main()
