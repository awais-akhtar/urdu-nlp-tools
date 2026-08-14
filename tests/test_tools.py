import unittest

from urdu_nlp import accuracy_score, augment_text, classification_report, contains_urdu, roman_to_urdu


class ToolTests(unittest.TestCase):
    def test_contains_urdu(self):
        self.assertTrue(contains_urdu("اردو"))
        self.assertFalse(contains_urdu("roman urdu"))

    def test_roman_to_urdu_returns_urdu_script(self):
        self.assertTrue(contains_urdu(roman_to_urdu("acha")))

    def test_augment_text_keeps_original(self):
        variants = augment_text("یہ فلم بہت اچھی تھی", random_deletions=1, seed=1)
        self.assertIn("یہ فلم بہت اچھی تھی", variants)

    def test_metrics(self):
        y_true = ["positive", "negative", "positive"]
        y_pred = ["positive", "positive", "positive"]
        report = classification_report(y_true, y_pred)
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 2 / 3)
        self.assertIn("macro_avg", report)


if __name__ == "__main__":
    unittest.main()
