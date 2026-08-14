
from __future__ import annotations

import re

_URDU_RE = re.compile(r"[\u0600-\u06ff]")
_ROMAN_WORD_RE = re.compile(r"[A-Za-z]+")

_ROMAN_CHUNKS: tuple[tuple[str, str], ...] = (
    ("kh", "خ"),
    ("gh", "غ"),
    ("ch", "چ"),
    ("sh", "ش"),
    ("ph", "ف"),
    ("th", "تھ"),
    ("bh", "بھ"),
    ("dh", "دھ"),
    ("aa", "ا"),
    ("ai", "ے"),
    ("ee", "ی"),
    ("oo", "و"),
    ("a", "ا"),
    ("b", "ب"),
    ("c", "ک"),
    ("d", "د"),
    ("e", "ے"),
    ("f", "ف"),
    ("g", "گ"),
    ("h", "ہ"),
    ("i", "ی"),
    ("j", "ج"),
    ("k", "ک"),
    ("l", "ل"),
    ("m", "م"),
    ("n", "ن"),
    ("o", "و"),
    ("p", "پ"),
    ("q", "ق"),
    ("r", "ر"),
    ("s", "س"),
    ("t", "ت"),
    ("u", "و"),
    ("v", "و"),
    ("w", "و"),
    ("x", "کس"),
    ("y", "ی"),
    ("z", "ز"),
)

_URDU_TO_ROMAN = {
    "ا": "a",
    "آ": "aa",
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ٹ": "t",
    "ث": "s",
    "ج": "j",
    "چ": "ch",
    "ح": "h",
    "خ": "kh",
    "د": "d",
    "ڈ": "d",
    "ذ": "z",
    "ر": "r",
    "ڑ": "r",
    "ز": "z",
    "ژ": "zh",
    "س": "s",
    "ش": "sh",
    "ص": "s",
    "ض": "z",
    "ط": "t",
    "ظ": "z",
    "ع": "a",
    "غ": "gh",
    "ف": "f",
    "ق": "q",
    "ک": "k",
    "گ": "g",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ں": "n",
    "و": "w",
    "ہ": "h",
    "ھ": "h",
    "ء": "",
    "ی": "y",
    "ے": "e",
}


def contains_urdu(text: str) -> bool:
    """Return True if the text contains Urdu/Arabic-script characters."""

    return bool(_URDU_RE.search(str(text)))


def is_urdu(text: str, *, threshold: float = 0.5) -> bool:
    """Estimate whether most letters in a string are Urdu-script characters."""

    value = str(text)
    letters = [char for char in value if char.isalpha()]
    if not letters:
        return False
    urdu_letters = sum(1 for char in letters if _URDU_RE.match(char))
    return (urdu_letters / len(letters)) >= threshold


def _roman_word_to_urdu(word: str) -> str:
    value = word.lower()
    output: list[str] = []
    index = 0
    while index < len(value):
        for roman, urdu in _ROMAN_CHUNKS:
            if value.startswith(roman, index):
                output.append(urdu)
                index += len(roman)
                break
        else:
            output.append(value[index])
            index += 1
    return "".join(output)


def roman_to_urdu(text: str) -> str:
    """Approximate Roman Urdu to Urdu-script transliteration."""

    return _ROMAN_WORD_RE.sub(lambda match: _roman_word_to_urdu(match.group(0)), str(text))


def urdu_to_roman(text: str) -> str:
    """Approximate Urdu-script to Roman Urdu transliteration."""

    return "".join(_URDU_TO_ROMAN.get(char, char) for char in str(text))
