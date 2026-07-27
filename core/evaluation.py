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
        probs = np.concatenate(self._probs)

        # Meke oznake
        if gt.shape == probs.shape:
            gt = np.argmax(gt, axis=1)

        preds = np.argmax(probs, axis=1)

        # Prostorne dimenzije
        if len(gt.shape) == 3:
            gt = gt.reshape(-1)
            preds = preds.reshape(-1)

            if len(probs.shape) == 4:
                probs = probs.transpose(0, 2, 3, 1).reshape(-1, probs.shape[1])
            else:
                probs = probs.reshape(-1)

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
        self._modified = False
        return metrics

    def reset(self):
        self._gt = []
        self._probs = []
        self._cache = None
        self._modified = False


if __name__ == "__main__":
    from pprint import pprint
    from lovely_numpy import lo

    np.random.seed(1414213)

    evaluator = Evaluator()

    probs = np.random.random((1, 1, 3, 3))
    probs = np.concatenate([probs, 1 - probs], axis=1)
    print("Input shape", probs.shape)

    gt = np.random.random((1, 1, 3, 3))
    gt = np.concatenate([gt, 1 - gt], axis=1)

    evaluator.update(gt, probs)
    result = evaluator.evaluate()
    pprint(result)
    print()

    np.random.seed(1414213)

    probs = np.random.random((5, 1))
    probs = np.concatenate([probs, 1 - probs], axis=1)
    print("Input shape", probs.shape)

    gt = np.array(
        [
            [0.9, 0.1],
            [0.75, 0.25],
            [1.0, 0.0],
            [0.1, 0.9],
            [0.1, 0.8],
        ]
    )

    evaluator.update(gt, probs)
    result = evaluator.evaluate()
    pprint(result)
    print()
