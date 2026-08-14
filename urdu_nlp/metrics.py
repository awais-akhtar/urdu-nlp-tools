"""Pure-Python metrics for sentiment classification."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Iterable

Label = Hashable


def _as_lists(y_true: Iterable[Label], y_pred: Iterable[Label]) -> tuple[list[Label], list[Label]]:
    true_values = list(y_true)
    pred_values = list(y_pred)
    if len(true_values) != len(pred_values):
        raise ValueError("y_true and y_pred must have the same length")
    return true_values, pred_values


def _labels(y_true: list[Label], y_pred: list[Label], labels: Iterable[Label] | None) -> list[Label]:
    if labels is not None:
        return list(labels)
    return list(dict.fromkeys([*y_true, *y_pred]))


def accuracy_score(y_true: Iterable[Label], y_pred: Iterable[Label]) -> float:
    """Return exact-match accuracy."""

    true_values, pred_values = _as_lists(y_true, y_pred)
    if not true_values:
        return 0.0
    return sum(1 for true, pred in zip(true_values, pred_values) if true == pred) / len(true_values)


def confusion_matrix(
    y_true: Iterable[Label],
    y_pred: Iterable[Label],
    *,
    labels: Iterable[Label] | None = None,
) -> dict[Label, dict[Label, int]]:
    """Return a nested dict indexed by true label then predicted label."""

    true_values, pred_values = _as_lists(y_true, y_pred)
    label_values = _labels(true_values, pred_values, labels)
    matrix = {true: {pred: 0 for pred in label_values} for true in label_values}
    for true, pred in zip(true_values, pred_values):
        if true not in matrix:
            matrix[true] = {label: 0 for label in label_values}
        if pred not in matrix[true]:
            matrix[true][pred] = 0
        matrix[true][pred] += 1
    return matrix


def precision_recall_f1(
    y_true: Iterable[Label],
    y_pred: Iterable[Label],
    *,
    labels: Iterable[Label] | None = None,
    average: str | None = "macro",
) -> dict[str, float] | dict[Label, dict[str, float]]:
    """Compute precision, recall, and F1."""

    true_values, pred_values = _as_lists(y_true, y_pred)
    label_values = _labels(true_values, pred_values, labels)
    matrix = confusion_matrix(true_values, pred_values, labels=label_values)
    support_counts = Counter(true_values)
    per_label: dict[Label, dict[str, float]] = {}

    for label in label_values:
        tp = matrix.get(label, {}).get(label, 0)
        fp = sum(matrix.get(other, {}).get(label, 0) for other in label_values if other != label)
        fn = sum(matrix.get(label, {}).get(other, 0) for other in label_values if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": float(support_counts[label]),
        }

    if average is None:
        return per_label

    if average == "macro":
        divisor = len(label_values) or 1
        return {
            metric: sum(values[metric] for values in per_label.values()) / divisor
            for metric in ("precision", "recall", "f1")
        }
    if average == "weighted":
        total = sum(support_counts.values()) or 1
        return {
            metric: sum(values[metric] * values["support"] for values in per_label.values()) / total
            for metric in ("precision", "recall", "f1")
        }
    if average == "micro":
        correct = sum(matrix.get(label, {}).get(label, 0) for label in label_values)
        total = len(true_values) or 1
        score = correct / total
        return {"precision": score, "recall": score, "f1": score}

    raise ValueError("average must be 'macro', 'weighted', 'micro', or None")


def classification_report(
    y_true: Iterable[Label],
    y_pred: Iterable[Label],
    *,
    labels: Iterable[Label] | None = None,
) -> dict[str, object]:
    """Return accuracy and per-label precision/recall/F1."""

    true_values, pred_values = _as_lists(y_true, y_pred)
    return {
        "accuracy": accuracy_score(true_values, pred_values),
        "macro_avg": precision_recall_f1(true_values, pred_values, labels=labels, average="macro"),
        "weighted_avg": precision_recall_f1(true_values, pred_values, labels=labels, average="weighted"),
        "per_label": precision_recall_f1(true_values, pred_values, labels=labels, average=None),
    }
