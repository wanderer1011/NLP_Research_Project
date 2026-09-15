"""
Evaluation utilities
"""

from dataclasses import dataclass


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
    false_negative_rate: float


def evaluate(
    y_true: list[int],
    y_pred: list[int],
) -> Metrics:
    """Compute binary classification metrics."""
    assert len(y_true) == len(y_pred)

    tp = tn = fp = fn = 0
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == 1 and pred_label == 1:
            tp += 1
        elif true_label == 0 and pred_label == 0:
            tn += 1
        elif true_label == 0 and pred_label == 1:
            fp += 1
        else:
            fn += 1

    total = len(y_true)
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
