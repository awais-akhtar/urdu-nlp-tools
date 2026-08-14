import unittest

from urdu_nlp import normalize_digits, normalize_urdu, remove_diacritics


class NormalizeTests(unittest.TestCase):
    def test_normalize_arabic_codepoints(self):
        self.assertEqual(normalize_urdu("كيا يہ ۱۲۳ ہے؟"), "کیا یہ 123 ہے؟")

    def test_remove_diacritics(self):
        self.assertEqual(remove_diacritics("بِہت"), "بہت")

    def test_urdu_digits(self):
        self.assertEqual(normalize_digits("Room 123", target="urdu"), "Room ۱۲۳")


if __name__ == "__main__":
    unittest.main()
