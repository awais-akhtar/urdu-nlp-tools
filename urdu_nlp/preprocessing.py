"""Reusable preprocessing utilities for Urdu and Roman Urdu datasets."""

from __future__ import annotations

import csv
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .normalize import normalize_urdu
from .transliteration import contains_urdu

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@\w+")
_HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_PUNCTUATION = set(string.punctuation) | set("،؛؟۔“”‘’«»")


@dataclass(frozen=True)
class DatasetRecord:
    """A labeled text row used for model training or evaluation."""

    text: str
    label: str


def _strip_punctuation(text: str) -> str:
    return "".join(" " if char in _PUNCTUATION else char for char in text)


def clean_text(
    text: str,
    *,
    lowercase_roman: bool = True,
    remove_urls: bool = True,
    remove_mentions: bool = True,
    keep_hashtag_text: bool = True,
    remove_punctuation: bool = False,
    remove_digits: bool = False,
) -> str:
    """Clean text while preserving Urdu characters."""

    value = str(text).replace("_", " ")
    if remove_urls:
        value = _URL_RE.sub(" ", value)
    if remove_mentions:
        value = _MENTION_RE.sub(" ", value)
    if keep_hashtag_text:
        value = _HASHTAG_RE.sub(r"\1", value)
    else:
        value = _HASHTAG_RE.sub(" ", value)

    if contains_urdu(value):
        value = normalize_urdu(value)
    elif lowercase_roman:
        value = value.lower()

    if remove_digits:
        value = re.sub(r"\d+", " ", value)
    if remove_punctuation:
        value = _strip_punctuation(value)

    return _WHITESPACE_RE.sub(" ", value).strip()


def tokenize_words(text: str) -> list[str]:
    """Tokenize Urdu or Roman Urdu text into word-like tokens."""

    return _TOKEN_RE.findall(clean_text(text, remove_punctuation=True))


def prepare_text(text: str, **clean_options: object) -> str:
    """Return normalized text suitable for vectorizers and rule-based scoring."""

    options = {"remove_punctuation": True}
    options.update(clean_options)
    return clean_text(text, **options)


def prepare_texts(texts: Iterable[str], **clean_options: object) -> list[str]:
    """Prepare many texts with the same options."""

    return [prepare_text(text, **clean_options) for text in texts]


def load_labeled_csv(
    path: str | Path,
    *,
    text_column: str = "review",
    label_column: str = "sentiment",
    encoding: str = "utf-8-sig",
    clean: bool = True,
) -> list[DatasetRecord]:
    """Load a CSV with text and sentiment columns."""

    records: list[DatasetRecord] = []
    with Path(path).open("r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        missing = {text_column, label_column} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        for row in reader:
            text = row.get(text_column, "")
            label = row.get(label_column, "")
            if text is None or label is None:
                continue
            text = prepare_text(text) if clean else str(text).strip()
            label = str(label).strip().lower()
            if text and label:
                records.append(DatasetRecord(text=text, label=label))
    return records
