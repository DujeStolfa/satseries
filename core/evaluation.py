from enum import StrEnum
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)


def _per_class_average_precision(gt, probs):
    return np.array(
        [average_precision_score(gt == i, probs[:, i]) for i in range(probs.shape[-1])]
    )


class EvaluationMetric(StrEnum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AVERAGE_PRECISION = "ap"


class Evaluator:
    def __init__(self):
        self._gt = []
        self._probs = []
        self._cache = None
        self._modified = False

    def update(self, target, probs):
        self._gt.append(target)
        self._probs.append(probs)
        self._modified = True

    def evaluate(self) -> Dict[EvaluationMetric, float | np.ndarray]:
        if self._cache and not self._modified:
            return self._cache

        gt = np.concatenate(self._gt)
        if len(gt.shape) == 2:
            gt = np.argmax(gt, axis=1)

        probs = np.concatenate(self._probs)
        preds = np.argmax(probs, axis=1)

        acc = accuracy_score(gt, preds)
        ap = _per_class_average_precision(gt, probs)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt, preds, average=None
        )

        metrics = {
            EvaluationMetric.ACCURACY: acc,
            EvaluationMetric.PRECISION: precision,
            EvaluationMetric.RECALL: recall,
            EvaluationMetric.F1: f1,
            EvaluationMetric.AVERAGE_PRECISION: ap,
        }
        self._cache = metrics
        self._modified = True
        return metrics

    def reset(self):
        self._gt = []
        self._probs = []
        self._cache = None
        self._modified = False
