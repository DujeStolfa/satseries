import math

import numpy as np
import torch.utils.data as data


class BalancedBatchSampler(data.BatchSampler):
    def __init__(
        self,
        positive_indices,
        negative_indices,
        batch_size,
        positive_ratio,
        drop_last=False,
        seed=None,
    ):
        self._positive_indices = positive_indices
        self._negative_indices = negative_indices
        self._batch_size = batch_size

        assert positive_ratio <= 1 and positive_ratio >= 0
        self._positive_ratio = positive_ratio
        self._positives_per_batch = math.ceil(batch_size * positive_ratio)
        self._negatives_per_batch = batch_size - self._positives_per_batch

        self._drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            batch_size=self._batch_size,
            positive_ratio=self._positive_ratio,
            drop_last=self._drop_last,
            seed=self._seed,
        )

    def __len__(self):
        count = len(self._positive_indices) / self._positives_per_batch

        if not self._drop_last:
            return math.ceil(count)

        return math.floor(count)

    def __iter__(self):
        batch_count = len(self)
        negative_count = batch_count * self._negatives_per_batch
        negatives = self._rng.choice(
            self._negative_indices,
            size=negative_count,
            replace=False,
        )
        positives = self._rng.permutation(self._positive_indices)

        positive_ptr = 0
        negative_ptr = 0

        for _ in range(batch_count):
            batch_pos = positives[
                positive_ptr : positive_ptr + self._positives_per_batch
            ]
            batch_neg = negatives[
                negative_ptr : negative_ptr + self._negatives_per_batch
            ]

            batch = np.concatenate([batch_pos, batch_neg])
            self._rng.shuffle(batch)

            yield batch.tolist()

            positive_ptr += self._positives_per_batch
            negative_ptr += self._negatives_per_batch
