
from __future__ import annotations

import random
from typing import Iterable, Sequence

from .normalize import normalize_urdu, remove_diacritics
from .preprocessing import prepare_text, tokenize_words


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    """Deduplicate strings while preserving their first-seen order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def drop_random_words(text: str, *, drop_probability: float = 0.12, seed: int | None = None) -> str:
    """Drop a small random subset of words."""

    if not 0 <= drop_probability < 1:
        raise ValueError("drop_probability must be in the range [0, 1)")
    rng = random.Random(seed)
    tokens = tokenize_words(text)
    if len(tokens) <= 1:
        return prepare_text(text)
    kept = [token for token in tokens if rng.random() >= drop_probability]
    return " ".join(kept or tokens[:1])


def swap_adjacent_words(text: str, *, swaps: int = 1, seed: int | None = None) -> str:
    """Swap adjacent word pairs to make a conservative noisy variant."""

    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    rng = random.Random(seed)
    tokens = tokenize_words(text)
    if len(tokens) < 2 or swaps == 0:
        return " ".join(tokens)
    for _ in range(swaps):
        index = rng.randrange(0, len(tokens) - 1)
        tokens[index], tokens[index + 1] = tokens[index + 1], tokens[index]
    return " ".join(tokens)


def augment_text(
    text: str,
    *,
    random_deletions: int = 0,
    random_swaps: int = 0,
    seed: int | None = None,
) -> list[str]:
    """Create conservative variants of one Urdu or Roman Urdu text."""

    base = str(text)
    variants = [
        base,
        normalize_urdu(base) if any("\u0600" <= char <= "\u06ff" for char in base) else prepare_text(base),
        remove_diacritics(base),
        prepare_text(base),
    ]
    for index in range(random_deletions):
        variants.append(drop_random_words(base, seed=None if seed is None else seed + index))
    for index in range(random_swaps):
        variants.append(swap_adjacent_words(base, seed=None if seed is None else seed + 1000 + index))
    return unique_preserve_order(variant for variant in variants if variant.strip())


def augment_dataset(
    texts: Sequence[str],
    labels: Sequence[str] | None = None,
    *,
    random_deletions: int = 1,
    random_swaps: int = 1,
    seed: int | None = None,
) -> list[str] | tuple[list[str], list[str]]:
    """Augment texts and optionally repeat matching labels."""

    augmented_texts: list[str] = []
    augmented_labels: list[str] = []
    if labels is not None and len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    for index, text in enumerate(texts):
        variants = augment_text(
            text,
            random_deletions=random_deletions,
            random_swaps=random_swaps,
            seed=None if seed is None else seed + index,
        )
        augmented_texts.extend(variants)
        if labels is not None:
            augmented_labels.extend([labels[index]] * len(variants))

    if labels is None:
        return augmented_texts
    return augmented_texts, augmented_labels
