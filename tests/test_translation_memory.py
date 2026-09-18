import json
import unittest

from translation_providers import TranslationProvider
from translate_text_test import collapse_duplicated_sentence, group_dialogue, merge_vertical_fragments


class FakeMemoryProvider(TranslationProvider):
    def __init__(self):
        self.prompt = ""

    async def _request(self, prompt, schema):
        self.prompt = prompt
        return json.dumps({
            "translations": [{"id": "bubble-001", "translation": "PAKAI SHADOW STEP!"}],
            "memory_updates": {"Shadow Step": "Shadow Step", "Jinwoo": "Jinwoo"},
        })


class RenumberingProvider(TranslationProvider):
    async def _request(self, prompt, schema):
        return json.dumps({"translations": [
            {"id": "1", "translation": "SATU"},
            {"id": "2", "translation": "DUA"},
        ]})


class EditorDropsCanonicalTermProvider(TranslationProvider):
    def __init__(self):
        self.calls = 0

    async def _request(self, prompt, schema):
        self.calls += 1
        translation = "Aura milikmu kuat." if self.calls % 2 else "Energi milikmu kuat."
        return json.dumps({"translations": [
            {"id": "bubble-009", "translation": translation, "style": "dialogue"},
        ]})


class TranslationMemoryTests(unittest.IsolatedAsyncioTestCase):
    def test_overlapping_vertical_fragments_in_one_bubble_are_merged(self):
        groups = [
            {
                "id": "bubble-007", "text": "THE MOMENT YOU MESS WITH",
                "bbox": {"x": 215, "y": 11903, "width": 494, "height": 114},
                "source_region_ids": ["a", "b"],
            },
            {
                "id": "bubble-008", "text": "SEO BEOM, THAT GUY WILL SHOW UP.",
                "bbox": {"x": 235, "y": 11961, "width": 456, "height": 161},
                "source_region_ids": ["c", "d", "e"],
            },
        ]
        merged = merge_vertical_fragments(groups)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            merged[0]["text"],
            "THE MOMENT YOU MESS WITH SEO BEOM, THAT GUY WILL SHOW UP.",
        )
        self.assertEqual(merged[0]["source_region_ids"], ["a", "b", "c", "d", "e"])

    def test_separate_narration_boxes_do_not_merge_via_accumulated_bbox(self):
        def row(identifier, text, x, y, width, height):
            return {
                "id": identifier, "text": text, "confidence": 0.99,
                "bbox": {"x": x, "y": y, "width": width, "height": height},
            }

        regions = [
            row("a", "GATHERS CHILDREN", 390, 1757, 311, 55),
            row("b", "FROM BOTH FAMILIES", 303, 1818, 467, 58),
            row("c", "MAIN AND BRANCH", 331, 1881, 411, 59),
            row("d", "FAMILIES...", 422, 1941, 236, 59),
            row("e", "ALONG WITH", 177, 2051, 287, 62),
            row("f", "ORPHANS AND SLAVES", 46, 2110, 523, 69),
        ]
        groups = group_dialogue(regions, 0)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["source_region_ids"], ["a", "b", "c", "d"])
        self.assertEqual(groups[1]["source_region_ids"], ["e", "f"])

    def test_identical_ocr_sentence_is_collapsed(self):
        self.assertEqual(
            collapse_duplicated_sentence("YOU'RE GOING TO DO IT? YOU'RE GOING TO DO IT?"),
            "YOU'RE GOING TO DO IT?",
        )

    def test_split_scanlator_promo_is_not_grouped_as_dialogue(self):
        regions = [
            {"id": "r1", "text": "Make sure to read this", "confidence": 0.99,
             "bbox": {"x": 78, "y": 238, "width": 250, "height": 40}},
            {"id": "r2", "text": "only at our website", "confidence": 0.99,
             "bbox": {"x": 90, "y": 278, "width": 230, "height": 40}},
        ]
        self.assertEqual(group_dialogue(regions, 0), [])

    def test_neighboring_words_on_same_line_are_consolidated(self):
        regions = [
            {"id": "a", "text": "AND", "confidence": .99, "bbox": {"x": 210, "y": 100, "width": 70, "height": 32}},
            {"id": "b", "text": "PICK", "confidence": .99, "bbox": {"x": 288, "y": 101, "width": 82, "height": 31}},
            {"id": "c", "text": "ONE", "confidence": .99, "bbox": {"x": 378, "y": 100, "width": 68, "height": 32}},
            {"id": "d", "text": "FLAVOR EACH?", "confidence": .99, "bbox": {"x": 451, "y": 100, "width": 205, "height": 35}},
        ]
        groups = group_dialogue(regions, 0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["text"], "AND PICK ONE FLAVOR EACH?")

    def test_stylised_low_confidence_second_line_stays_in_same_balloon(self):
        regions = [
            {"id": "a", "text": "YOU LITTLE", "confidence": .99,
             "bbox": {"x": 125, "y": 3247, "width": 458, "height": 146}},
            {"id": "b", "text": "SHIT...!!", "confidence": .725,
             "bbox": {"x": 195, "y": 3343, "width": 332, "height": 167}},
        ]
        groups = group_dialogue(regions, 0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["source_region_ids"], ["a", "b"])
        self.assertIn("SHIT", groups[0]["text"])

    def test_overlapping_tile_duplicate_with_damaged_letter_is_removed(self):
        regions = [
            {"id": "good", "text": "HEY!!", "confidence": .99,
             "bbox": {"x": 409, "y": 7158, "width": 189, "height": 81}},
            {"id": "tile", "text": "rEY!!", "confidence": .77,
             "bbox": {"x": 410, "y": 7159, "width": 187, "height": 80}},
        ]
        groups = group_dialogue(regions, 0)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["text"], "HEY!!")
        self.assertEqual(groups[0]["source_region_ids"], ["good"])

    def test_thin_decorative_strip_is_not_dialogue(self):
        regions = [{
            "id": "ui", "text": "SYSTEM DECORATIVE LABEL", "confidence": .99,
            "bbox": {"x": 66, "y": 4430, "width": 145, "height": 13},
        }]
        self.assertEqual(group_dialogue(regions, 0), [])

    async def test_memory_is_in_prompt_and_new_terms_are_collected(self):
        provider = FakeMemoryProvider()
        rows = await provider.translate(
            [{"id": "bubble-001", "text": "Use Shadow Step!"}],
            {"Iron Body": "Iron Body"},
        )
        self.assertEqual(rows[0]["translation"], "PAKAI SHADOW STEP!")
        self.assertIn('"Iron Body": "Iron Body"', provider.prompt)
        self.assertIn("Never translate a named skill literally", provider.prompt)
        self.assertEqual(provider.memory_updates["Shadow Step"], "Shadow Step")

    async def test_provider_ids_are_repaired_when_row_count_and_order_match(self):
        rows = await RenumberingProvider().translate([
            {"id": "bubble-004", "text": "one"},
            {"id": "bubble-009", "text": "two"},
        ], {})
        self.assertEqual([row["id"] for row in rows], ["bubble-004", "bubble-009"])

    async def test_glossary_safe_draft_survives_editor_regression(self):
        provider = EditorDropsCanonicalTermProvider()
        rows = await provider.translate(
            [{"id": "bubble-009", "text": "YOUR AURA IS STRONG."}],
            {"Aura": "Aura"},
        )
        self.assertEqual(rows[0]["translation"], "Aura milikmu kuat.")
        self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
