import unittest
from pathlib import Path

from language_memory import restore_aliases
from pronoun_policy import normalize_pronouns


class PronounDefaultTests(unittest.TestCase):
    """Tanpa policy judul, hanya drift slang yang diperbaiki."""

    def test_slang_dinormalkan_ke_aku_kamu(self):
        self.assertEqual(normalize_pronouns("Gue nggak tahu, lo aja."), "Aku nggak tahu, kamu aja.")

    def test_formal_saya_anda_dibiarkan(self):
        self.assertEqual(normalize_pronouns("Saya menemui Anda."), "Saya menemui Anda.")

    def test_policy_locked_menormalkan_semua(self):
        policy = {"pronouns": "aku/kamu", "locked": True}
        self.assertEqual(normalize_pronouns("Saya dan kau, bukan gue.", policy),
                         "Aku dan kamu, bukan aku.")

    def test_allow_slang_dihormati(self):
        policy = {"pronouns": "aku/kamu", "locked": True, "allow_slang": True}
        self.assertIn("gue", normalize_pronouns("gue pergi", policy).lower())


class AliasTests(unittest.TestCase):
    def test_alias_istilah_dikembalikan(self):
        self.assertEqual(restore_aliases("Dia Penyihir Es.", {"Penyihir Es": "Ice Witch"}),
                         "Dia Ice Witch.")

    def test_alias_tidak_memotong_kata_lain(self):
        self.assertEqual(restore_aliases("Torii tetap.", {"Tori": "TOURI"}), "Torii tetap.")

    def test_alias_tanpa_entri(self):
        self.assertEqual(restore_aliases("Teks apa adanya.", {}), "Teks apa adanya.")


if __name__ == "__main__":
    unittest.main()


class AddressNormalizationTests(unittest.TestCase):
    """Honorifik untuk satu nama tidak boleh berubah-ubah antar halaman."""

    POLICY = {"address": {"Mr. Takeru": "Pak Takeru"}}

    def test_varian_menjadi_kanonik(self):
        from language_memory import normalize_address_terms
        for translated in ("Mas Takeru!", "Tuan Takeru!", "Takeru-san!", "Bapak Takeru!"):
            self.assertEqual(
                normalize_address_terms(translated, "Mr. Takeru!", self.POLICY),
                "Pak Takeru!",
            )

    def test_huruf_besar_dipertahankan(self):
        from language_memory import normalize_address_terms
        self.assertEqual(
            normalize_address_terms("TUAN TAKERU!", "NOW, MR. TAKERU!", self.POLICY),
            "PAK TAKERU!",
        )

    def test_nama_lain_tidak_ikut(self):
        from language_memory import normalize_address_terms
        self.assertEqual(
            normalize_address_terms("Mas Kuon datang.", "Mr. Kuon is here.", self.POLICY),
            "Tuan Kuon datang.",
        )

    def test_generik_tanpa_policy_judul(self):
        from language_memory import normalize_address_terms
        self.assertEqual(normalize_address_terms("Mas Kenji datang.", "Mr. Kenji is here.", None),
                         "Tuan Kenji datang.")
        self.assertEqual(normalize_address_terms("Nona Ayu menunggu.", "Miss Ayu is waiting.", None),
                         "Nona Ayu menunggu.")


class ChapterContextTests(unittest.TestCase):
    def test_konteks_menyimpan_sapaan_dan_baris(self):
        import tempfile
        from chapter_context import load, update
        path = Path(tempfile.mkdtemp()) / "ctx.json"
        update(path, 1, [{"translation": "Pak Takeru! Aku kangen!"}])
        update(path, 2, [{"translation": "TENANG DULU, PAK TAKERU!"}])
        ctx = load(path)
        self.assertEqual(ctx["address"], ["Pak Takeru"])
        self.assertEqual(ctx["pages"], 2)
        self.assertGreaterEqual(len(ctx["lines"]), 2)

    def test_sapaan_mapan_dipaksakan(self):
        from language_memory import enforce_established_address
        self.assertEqual(enforce_established_address("Mas Takeru!", ["Pak Takeru"]), "Pak Takeru!")
        self.assertEqual(enforce_established_address("TUAN TAKERU!", ["Pak Takeru"]), "PAK TAKERU!")
        self.assertEqual(enforce_established_address("Mas Kuon!", ["Pak Takeru"]), "Mas Kuon!")


class LearnedAddressTests(unittest.TestCase):
    def test_peta_nama_hasil_belajar(self):
        from language_memory import normalize_address_terms
        policy = {"address_names": {"Takeru": "Pak Takeru"}}
        self.assertEqual(normalize_address_terms("Mas Takeru!", "Mr. Takeru!", policy), "Pak Takeru!")
        self.assertEqual(normalize_address_terms("TUAN TAKERU!", "MR. TAKERU!", policy), "PAK TAKERU!")

    def test_belajar_dari_halaman_selesai(self):
        import json
        import tempfile
        import queue_worker
        root = Path(tempfile.mkdtemp())
        page = root / "metadata/page-001"
        page.mkdir(parents=True)
        (page / "translations.json").write_text(json.dumps({
            "groups": [
                {"text": "Mr. Takeru! I miss him!", "translation": "Pak Takeru! Aku kangen!"},
                {"text": "Mr. Kuon is here. He arrived.", "translation": "Tuan Kuon datang."},
            ]
        }))
        memory_dir = Path(queue_worker.__file__).resolve().parent / "data/memory"
        before = None
        target = memory_dir / "zz-test-learn/editorial-policy.json"
        if target.exists():
            before = target.read_text()
        try:
            queue_worker.learn_chapter_addresses(root, {"title_slug": "zz-test-learn"})
            data = json.loads(target.read_text())
            self.assertEqual(data["address_names"]["Takeru"], "Pak Takeru")
            self.assertEqual(data["address_names"]["Kuon"], "Tuan Kuon")
        finally:
            if before is None:
                target.unlink(missing_ok=True)
                target.parent.rmdir()
            else:
                target.write_text(before)
