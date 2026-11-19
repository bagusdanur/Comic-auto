"""Regresi bug gender honorifik: gelar/kata sifat bukan nama, verifikasi sumber,
dan jangan timpa memori yang sudah mapan."""
import json
import tempfile
import unittest
from pathlib import Path

from chapter_context import iter_address, NON_NAME_TOKENS
from language_memory import (
    normalize_address_terms,
    enforce_established_address,
    honorific_gender,
    source_gender,
)
from queue_worker import learn_chapter_addresses


class IterAddressTests(unittest.TestCase):
    def test_gelar_dan_kata_sifat_dilewati(self):
        for text in ("Bu Guru sudah datang.", "Pak Guru, saya bertanya.",
                     "Tuan Muda Skarab.", "Pak Tua itu ayahnya.", "Nona Muda itu cantik."):
            names = [name for _, name in iter_address(text)]
            self.assertNotIn("Guru", names)
            self.assertNotIn("Muda", names)
            self.assertNotIn("Tua", names)

    def test_nama_diri_tetap_tertangkap(self):
        self.assertEqual(list(iter_address("Bu Aina cantik.")), [("Bu", "Aina")])
        self.assertEqual(list(iter_address("Tuan Takeru datang.")), [("Tuan", "Takeru")])
        self.assertEqual(list(iter_address("Tuan Muda Skarab.")), [("Tuan Muda", "Skarab")])

    def test_blacklist_memuat_token_kritis(self):
        for token in ("guru", "muda", "tua", "dokter", "kapten"):
            self.assertIn(token, NON_NAME_TOKENS)


class GenderHelperTests(unittest.TestCase):
    def test_honorific_gender(self):
        self.assertEqual(honorific_gender("Pak"), "male")
        self.assertEqual(honorific_gender("Tuan Muda"), "male")
        self.assertEqual(honorific_gender("Bu"), "female")
        self.assertEqual(honorific_gender("Nona"), "female")
        self.assertEqual(honorific_gender("Kak"), "neutral")

    def test_source_gender(self):
        self.assertEqual(source_gender("HE IS ANGRY"), "male")
        self.assertEqual(source_gender("SHE IS ANGRY"), "female")
        self.assertIsNone(source_gender("THE TEACHER IS ANGRY"))
        self.assertIsNone(source_gender("HE AND SHE ARGUED"))


class NormalizeNoLongerForcesWrongGenderTests(unittest.TestCase):
    def test_pak_guru_tidak_dipaksa_jadi_bu(self):
        out = normalize_address_terms("Pak Guru, saya mau bertanya.",
                                      "TEACHER, I HAVE A QUESTION.", {"address_names": {}})
        self.assertEqual(out, "Pak Guru, saya mau bertanya.")

    def test_tuan_muda_tidak_dipaksa_jadi_nona(self):
        out = normalize_address_terms("Tuan Muda Skarab.", "YOUNG MASTER SKARAB.",
                                      {"address_names": {}})
        self.assertEqual(out, "Tuan Muda Skarab.")

    def test_nama_perempuan_tetap_konsisten(self):
        out = normalize_address_terms("Bu Aina sudah pulang.", "SHE HAS GONE HOME.",
                                      {"address_names": {"Aina": "Nona Aina"}})
        self.assertEqual(out, "Nona Aina sudah pulang.")

    def test_generic_mr_guru_dilewati(self):
        out = normalize_address_terms("Guru itu marah.", "Mr. Guru is angry.",
                                      {"address_names": {}})
        self.assertEqual(out, "Guru itu marah.")


class LearnChapterAddressesTests(unittest.TestCase):
    def _job(self, tmp, groups):
        root = Path(tmp)
        page = root / "metadata" / "page-001"
        page.mkdir(parents=True)
        (page / "translations.json").write_text(json.dumps({"groups": groups}), encoding="utf-8")
        return root

    def test_gender_salah_tidak_dipelajari(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._job(tmp, [
                {"text": "HE IS THE TEACHER.", "translation": "Bu Guru itu marah."},
            ])
            learn_chapter_addresses(root, {"title_slug": "unit-test-gender"})
            policy = Path(__file__).resolve().parents[1] / "data/memory/unit-test-gender/editorial-policy.json"
            self.assertFalse(policy.exists(), "gender bertentangan tidak boleh terkunci")

    def test_gender_benar_dipelajari(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._job(tmp, [
                {"text": "SHE IS KIND.", "translation": "Bu Aina baik hati."},
            ])
            learn_chapter_addresses(root, {"title_slug": "unit-test-gender-ok"})
            policy = Path(__file__).resolve().parents[1] / "data/memory/unit-test-gender-ok/editorial-policy.json"
            self.assertTrue(policy.exists())
            data = json.loads(policy.read_text())
            self.assertEqual(data["address_names"]["Aina"], "Bu Aina")
            policy.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.rmtree(policy.parent, ignore_errors=True)

    def test_memori_mapan_tidak_ditimpa(self):
        slug = "unit-test-no-overwrite"
        policy_dir = Path(__file__).resolve().parents[1] / "data/memory" / slug
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy = policy_dir / "editorial-policy.json"
        policy.write_text(json.dumps({"address_names": {"Aina": "Nona Aina"}}), encoding="utf-8")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = self._job(tmp, [
                    {"text": "SHE IS KIND.", "translation": "Bu Aina baik hati."},
                ])
                learn_chapter_addresses(root, {"title_slug": slug})
            data = json.loads(policy.read_text())
            self.assertEqual(data["address_names"]["Aina"], "Nona Aina",
                             "bentuk mapan tidak boleh dibalik oleh chapter baru")
        finally:
            import shutil
            shutil.rmtree(policy_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
