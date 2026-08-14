"""Small, practical tools for Urdu and Roman Urdu NLP."""

from .augmentation import augment_text
from .metrics import accuracy_score, classification_report, confusion_matrix
from .normalize import normalize_digits, normalize_urdu, remove_diacritics
from .preprocessing import DatasetRecord, clean_text, load_labeled_csv, prepare_text, tokenize_words
from .sentiment import SentimentAnalyzer, SentimentResult
from .transliteration import contains_urdu, is_urdu, roman_to_urdu, urdu_to_roman

__version__ = "0.1.0"

__all__ = [
    "DatasetRecord",
    "SentimentAnalyzer",
    "SentimentResult",
    "accuracy_score",
    "augment_text",
    "classification_report",
    "clean_text",
    "confusion_matrix",
    "contains_urdu",
    "is_urdu",
    "load_labeled_csv",
    "normalize_digits",
    "normalize_urdu",
    "prepare_text",
    "remove_diacritics",
    "roman_to_urdu",
    "tokenize_words",
    "urdu_to_roman",
]
