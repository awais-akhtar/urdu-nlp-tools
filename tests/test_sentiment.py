import tempfile
import unittest
from pathlib import Path

from urdu_nlp import SentimentAnalyzer


class SentimentTests(unittest.TestCase):
    def test_default_positive_urdu_prediction(self):
        analyzer = SentimentAnalyzer()
        self.assertEqual(analyzer.predict("یہ فلم بہت اچھی تھی"), "positive")

    def test_default_negative_roman_prediction(self):
        analyzer = SentimentAnalyzer()
        self.assertEqual(analyzer.predict("movie bohat kharab thi"), "negative")

    def test_batch_prediction(self):
        analyzer = SentimentAnalyzer()
        self.assertEqual(
            analyzer.predict(["یہ فلم اچھی تھی", "ye movie fazool thi"]),
            ["positive", "negative"],
        )

    def test_train_save_and_load_when_sklearn_available(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is not installed")

        analyzer = SentimentAnalyzer().fit(
            ["bohat acha", "bohat kharab", "یہ اچھی ہے", "یہ خراب ہے"],
            ["positive", "negative", "positive", "negative"],
            max_features=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            analyzer.save(path)
            loaded = SentimentAnalyzer.from_file(path)
            self.assertIn(loaded.predict("bohat acha"), {"positive", "negative"})


if __name__ == "__main__":
    unittest.main()
