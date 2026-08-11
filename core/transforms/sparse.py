import torch

from core.datasets.types import SparseSeriesDatasetSample
from core.transforms.base import Transform


class BatchRandomTemporalTruncate(Transform):
    def __init__(self, start_month, end_month):
        self._start_month = start_month
        self._end_month = end_month

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["start_month"] = self._start_month
        cfg["end_month"] = self._end_month
        return cfg

    def __call__(self, data: SparseSeriesDatasetSample) -> SparseSeriesDatasetSample:
        b, _, _ = data.series.shape

        new_end_months = torch.randint(
            self._start_month, self._end_month + 1, size=(b, 1)
        )
        new_ignore_mask = data.timesteps[..., 1] > new_end_months

        data.ignore_mask = data.ignore_mask | new_ignore_mask
        return data
