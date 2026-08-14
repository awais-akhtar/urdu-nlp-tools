
from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .preprocessing import prepare_text, prepare_texts, tokenize_words

POSITIVE_WORDS = {
    "اچھا",
    "اچھی",
    "اچھے",
    "بہتر",
    "بہترین",
    "اعلی",
    "زبردست",
    "شاندار",
    "کمال",
    "لاجواب",
    "خوبصورت",
    "پسند",
    "محبت",
    "خوش",
    "مزہ",
    "acha",
    "achi",
    "achha",
    "behtar",
    "behtareen",
    "zabardast",
    "shandar",
    "kamal",
    "lajawab",
    "pasand",
    "khush",
    "good",
    "great",
    "excellent",
    "best",
    "love",
}

NEGATIVE_WORDS = {
    "برا",
    "بری",
    "برے",
    "خراب",
    "بدترین",
    "بیکار",
    "فضول",
    "ناکام",
    "کمزور",
    "نفرت",
    "غصہ",
    "مایوس",
    "افسوس",
    "مسئلہ",
    "ناپسند",
    "bura",
    "buri",
    "kharab",
    "bekar",
    "fazool",
    "nafrat",
    "ghussa",
    "mayus",
    "masla",
    "bad",
    "poor",
    "worst",
    "hate",
}

NEGATORS = {"نہیں", "نہ", "مت", "na", "nahi", "nahin", "not", "no"}
DEFAULT_LABELS = ("negative", "neutral", "positive")


@dataclass(frozen=True)
class SentimentResult:
    """A single sentiment prediction."""

    label: str
    score: float
    confidence: float
    scores: Mapping[str, float]


def _softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


class SentimentAnalyzer:
    """Predict Urdu or Roman Urdu sentiment.

    Without a trained model this class uses a small Urdu/Roman Urdu lexicon so
    examples work immediately. For production use, call ``fit`` with your own
    labeled dataset and save the model artifact.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        positive_words: Iterable[str] | None = None,
        negative_words: Iterable[str] | None = None,
    ) -> None:
        self.positive_words = {prepare_text(word) for word in (positive_words or POSITIVE_WORDS)}
        self.negative_words = {prepare_text(word) for word in (negative_words or NEGATIVE_WORDS)}
        self.negators = {prepare_text(word) for word in NEGATORS}
        self.vectorizer: Any | None = None
        self.classifier: Any | None = None
        self.label_encoder: Any | None = None
        self.labels = DEFAULT_LABELS
        if model_path is not None:
            self.load(model_path)

    @property
    def is_trained(self) -> bool:
        """Return True when a vectorizer and classifier are loaded."""

        return self.vectorizer is not None and self.classifier is not None

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        max_features: int = 50000,
        ngram_range: tuple[int, int] = (1, 2),
        max_iter: int = 1000,
    ) -> "SentimentAnalyzer":
        """Train a TF-IDF + LinearSVC model from labeled texts."""

        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("at least one training example is required")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            from sklearn.svm import LinearSVC
        except ImportError as exc:
            raise ImportError("Install training dependencies with: pip install 'urdu-nlp-tools[train]'") from exc

        prepared = prepare_texts(texts)
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform([str(label).lower().strip() for label in labels])
        self.labels = tuple(str(label) for label in self.label_encoder.classes_)
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)
        vectors = self.vectorizer.fit_transform(prepared)
        self.classifier = LinearSVC(max_iter=max_iter)
        self.classifier.fit(vectors, encoded_labels)
        return self

    def analyze(self, text: str) -> SentimentResult:
        """Return label, score, confidence, and per-label scores for one text."""

        if self.is_trained:
            return self._analyze_with_model(text)
        return self._analyze_with_lexicon(text)

    def analyze_many(self, texts: Iterable[str]) -> list[SentimentResult]:
        """Analyze many texts."""

        return [self.analyze(text) for text in texts]

    def predict(self, texts: str | Iterable[str]) -> str | list[str]:
        """Predict a label for one text or many texts."""

        if isinstance(texts, str):
            return self.analyze(texts).label
        return [result.label for result in self.analyze_many(texts)]

    def save(self, path: str | Path) -> None:
        """Save the trained analyzer to disk."""

        if not self.is_trained:
            raise ValueError("fit or load a trained model before saving")
        payload = {
            "kind": "tfidf-linear-svc",
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "label_encoder": self.label_encoder,
            "labels": self.labels,
        }
        with Path(path).open("wb") as file:
            pickle.dump(payload, file)

    def load(self, path: str | Path) -> "SentimentAnalyzer":
        """Load a model saved by ``SentimentAnalyzer.save``."""

        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict):
            raise ValueError("model artifact must be a dictionary")
        self.vectorizer = payload.get("vectorizer")
        self.classifier = payload.get("classifier") or payload.get("model")
        self.label_encoder = payload.get("label_encoder")
        labels = payload.get("labels")
        if labels is not None:
            self.labels = tuple(str(label) for label in labels)
        elif self.label_encoder is not None:
            self.labels = tuple(str(label) for label in self.label_encoder.classes_)
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("model artifact must include a vectorizer and classifier")
        return self

    @classmethod
    def from_file(cls, path: str | Path) -> "SentimentAnalyzer":
        """Create an analyzer from a saved model artifact."""

        return cls(model_path=path)

    def _analyze_with_model(self, text: str) -> SentimentResult:
        prepared = prepare_text(text)
        vectors = self.vectorizer.transform([prepared])
        encoded = self.classifier.predict(vectors)[0]
        if self.label_encoder is not None:
            label = str(self.label_encoder.inverse_transform([encoded])[0])
        else:
            label = str(encoded)
        scores = self._decision_scores(vectors, label)
        confidence = scores.get(label, max(scores.values()) if scores else 1.0)
        return SentimentResult(label=label, score=confidence, confidence=confidence, scores=scores)

    def _decision_scores(self, vectors: Any, label: str) -> dict[str, float]:
        if not hasattr(self.classifier, "decision_function"):
            return {candidate: float(candidate == label) for candidate in self.labels}

        raw = self.classifier.decision_function(vectors)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        row = raw[0] if isinstance(raw, list) else raw
        if not isinstance(row, list):
            value = float(row)
            row = [-value, value] if len(self.labels) == 2 else [value for _ in self.labels]
        probabilities = _softmax([float(value) for value in row])
        return dict(zip(self.labels, probabilities))

    def _analyze_with_lexicon(self, text: str) -> SentimentResult:
        prepared = prepare_text(text)
        tokens = tokenize_words(prepared)
        token_set = set(tokens)
        positive = self._count_terms(prepared, token_set, self.positive_words)
        negative = self._count_terms(prepared, token_set, self.negative_words)

        for previous, current in zip(tokens, tokens[1:]):
            if previous in self.negators and current in self.positive_words:
                positive -= 1
                negative += 1
            elif previous in self.negators and current in self.negative_words:
                negative -= 1
                positive += 1

        raw_score = positive - negative
        total = max(positive + negative, 0)
        if raw_score > 0:
            label = "positive"
        elif raw_score < 0:
            label = "negative"
        else:
            label = "neutral"

        confidence = 0.5 if total == 0 else min(0.95, 0.5 + abs(raw_score) / (2 * max(total, 1)))
        scores = self._heuristic_scores(label, confidence)
        return SentimentResult(label=label, score=float(raw_score), confidence=confidence, scores=scores)

    @staticmethod
    def _count_terms(text: str, token_set: set[str], terms: set[str]) -> int:
        count = 0
        for term in terms:
            if not term:
                continue
            if " " in term and term in text:
                count += 2
            elif term in token_set:
                count += 1
        return count

    @staticmethod
    def _heuristic_scores(label: str, confidence: float) -> dict[str, float]:
        if label == "neutral":
            return {"negative": 0.25, "neutral": 0.5, "positive": 0.25}
        remaining = max(0.0, 1.0 - confidence)
        scores = {"negative": remaining / 2, "neutral": remaining / 2, "positive": remaining / 2}
        scores[label] = confidence
        return scores


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for quick predictions."""

    parser = argparse.ArgumentParser(description="Predict Urdu or Roman Urdu sentiment.")
    parser.add_argument("text", nargs="*", help="Text to analyze. If omitted, stdin is read.")
    parser.add_argument("--model", help="Path to a model saved with SentimentAnalyzer.save().")
    args = parser.parse_args(argv)

    text = " ".join(args.text)
    if not text:
        text = input().strip()
    analyzer = SentimentAnalyzer(args.model)
    result = analyzer.analyze(text)
    print(f"{result.label}\tconfidence={result.confidence:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
