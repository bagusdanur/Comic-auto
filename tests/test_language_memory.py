import unittest
from language_memory import terminology_errors,grounded_updates,LANGUAGE_POLICY
class LanguageMemoryTests(unittest.TestCase):
    def test_plural_and_case(self):
        self.assertEqual(terminology_errors([{'id':'a','text':'WYVERNS!'}],[{'translation':'Wyvern!'}],{'Wyvern':'Wyvern'}),[])
    def test_wrong_name_rejected(self):
        self.assertTrue(terminology_errors([{'id':'a','text':'Erica!'}],[{'translation':'Eric!'}],{'Erica':'Erica'}))
    def test_longest_term_wins(self):
        self.assertEqual(terminology_errors([{'id':'a','text':'Merchant Guild'}],[{'translation':'serikat pedagang'}],{'Merchant':'Merchant','Merchant Guild':'serikat pedagang'}),[])
    def test_substrings_are_not_names(self):
        self.assertEqual(terminology_errors([{'id':'a','text':'America'}],[{'translation':'Amerika'}],{'Eric':'Eric'}),[])
    def test_unrelated_and_existing_updates_not_learned(self):
        self.assertEqual(grounded_updates({'Erica':'Erika','Elsewhere':'Tempat lain','Aura':'Aura'},[{'text':'Erica uses Aura'}],{'Erica':'Erica'}),{'Aura':'Aura'})
    def test_policy_protects_meaning(self):
        self.assertIn('comparison direction',LANGUAGE_POLICY)
        self.assertIn('Approved terminology wins',LANGUAGE_POLICY)
