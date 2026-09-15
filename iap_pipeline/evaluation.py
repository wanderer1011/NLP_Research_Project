"""
Evaluation utilities — computes standard metrics comparing IAP predictions
against the ground-truth MentalManip labels.
"""

from __future__ import annotations

from dataclasses import dataclass

from classifier import ClassificationResult
from data_loader import Dialogue


@dataclass
class Metrics:
    total: int
    tp: int  # true positives (correctly predicted manipulative)
    tn: int  # true negatives
    fp: int  # false positives
    fn: int  # false negatives
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_negative_rate: float  # key metric for IAP: should be low


def evaluate(
    dialogues: list[Dialogue],
    predictions: list[ClassificationResult],
) -> Metrics:
    """Compute binary classification metrics."""
    assert len(dialogues) == len(predictions)

    tp = tn = fp = fn = 0
    for d, p in zip(dialogues, predictions):
        if d.label == 1 and p.manipulative == 1:
            tp += 1
        elif d.label == 0 and p.manipulative == 0:
            tn += 1
        elif d.label == 0 and p.manipulative == 1:
            fp += 1
        else:
            fn += 1

    total = len(dialogues)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0

    return Metrics(
        total=total,
        tp=tp, tn=tn, fp=fp, fn=fn,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        false_negative_rate=fnr,
    )
