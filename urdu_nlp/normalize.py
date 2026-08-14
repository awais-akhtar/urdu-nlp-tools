"""Normalization helpers for Urdu text."""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_WHITESPACE_RE = re.compile(r"\s+")

_ARABIC_TO_URDU = str.maketrans(
    {
        "ك": "ک",
        "ڪ": "ک",
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ة": "ہ",
        "ۀ": "ہ",
        "ھ": "ہ",
        "ؤ": "و",
        "إ": "ا",
        "أ": "ا",
        "ٱ": "ا",
    }
)

_ZERO_WIDTH = {
    ord("\u200b"): None,
    ord("\u200c"): None,
    ord("\u200d"): None,
    ord("\ufeff"): None,
}

_URDU_TO_ASCII_DIGITS = {
    **{ord(ch): str(i) for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹")},
    **{ord(ch): str(i) for i, ch in enumerate("٠١٢٣٤٥٦٧٨٩")},
}
_ASCII_TO_URDU_DIGITS = {ord(str(i)): ch for i, ch in enumerate("۰۱۲۳۴۵۶۷۸۹")}


def remove_diacritics(text: str) -> str:
    """Remove Urdu/Arabic combining marks from text."""

    return _DIACRITICS_RE.sub("", str(text))


def normalize_digits(text: str, target: str = "ascii") -> str:
    """Normalize Urdu, Arabic, or ASCII digits."""

    value = str(text)
    if target == "ascii":
        return value.translate(_URDU_TO_ASCII_DIGITS)
    if target == "urdu":
        return value.translate(_URDU_TO_ASCII_DIGITS).translate(_ASCII_TO_URDU_DIGITS)
    raise ValueError("target must be 'ascii' or 'urdu'")


def normalize_urdu(
    text: str,
    *,
    keep_diacritics: bool = False,
    digits: str | None = "ascii",
) -> str:
    """Normalize common Urdu and Arabic code-point variants.

    The function intentionally stays conservative: it normalizes glyph variants,
    optional diacritics, digits, zero-width characters, and repeated whitespace
    without stemming or changing word order.
    """

    value = unicodedata.normalize("NFKC", str(text))
    value = value.translate(_ZERO_WIDTH)
    value = value.translate(_ARABIC_TO_URDU)
    if not keep_diacritics:
        value = remove_diacritics(value)
    if digits is not None:
        value = normalize_digits(value, target=digits)
    return _WHITESPACE_RE.sub(" ", value).strip()
