from typing import List

import numpy as np
import torch

from core.datasets import PrestoDatasetSample, MultimodalDatasetSample
from core.datasets.base import Modality
from core.transforms.base import Transform


class ApplyToModality(Transform):
    def __init__(self, transform: Transform, modalities: Modality | List[Modality]):
        self._transform = transform
        self._modalities = modalities if isinstance(modalities, list) else [modalities]

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["transform"] = self._transform.config_dict
        cfg["modalities"] = [m.value for m in self._modalities]
        return cfg

    def __call__(self, data: MultimodalDatasetSample):
        for m in self._modalities:
            if m not in data.modalities:
                raise ValueError(f"Dataset sample doesn't contain modality {m.value}")

            data.modalities[m] = self._transform(data.modalities[m])

        return data


class ToTensor(Transform):
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()

        x.to_tensor()
        return x


class BiasedRandomCrop(Transform):
    def __init__(self, width, height, background=-1, seed=None):
        self._crop_width = width
        self._crop_height = height
        self._background = background
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["width"] = self._crop_width
        cfg["height"] = self._crop_height
        cfg["background"] = self._background
        cfg["seed"] = self._seed
        return cfg

    def __call__(self, x: MultimodalDatasetSample) -> MultimodalDatasetSample:
        target_data = x.target.numpy() if hasattr(x.target, "numpy") else x.target
        valid_pixels = np.argwhere(target_data != self._background)

        img_height, img_width = x.target.shape[-2:]

        if len(valid_pixels) == 0:
            top = self._rng.integers(0, img_height - self._crop_height + 1)
            left = self._rng.integers(0, img_width - self._crop_width + 1)

        else:
            valid_y, valid_x = valid_pixels[self._rng.integers(0, len(valid_pixels))]

            low_y = max(0, valid_y - self._crop_height + 1)
            high_y = min(img_height - self._crop_height, valid_y)

            low_x = max(0, valid_x - self._crop_width + 1)
            high_x = min(img_width - self._crop_width, valid_x)

            top = self._rng.integers(low_y, high_y + 1) if low_y < high_y else low_y
            left = self._rng.integers(low_x, high_x + 1) if low_x < high_x else low_x

        for ts in x.modalities.values():
            ts.images = ts.images[
                ..., top : top + self._crop_height, left : left + self._crop_width
            ]

        x.target = x.target[
            ..., top : top + self._crop_height, left : left + self._crop_width
        ]

        return x


class ToPrestoFormat(Transform):
    def __init__(self, month_start):
        self._month_start = month_start
        self._s1_out_channels = [0, 1]
        self._s2_out_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16]
        self._s2_in_channels = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["month_start"] = self._month_start
        return cfg

    def __call__(self, x: MultimodalDatasetSample) -> PrestoDatasetSample:
        if len(x.modalities) == 0:
            raise ValueError("Received empty MultimodalDatasetSample")

        ts = list(x.modalities.values())[0]
        ymdh = ts.timesteps  # max?
        b, t, _ = ts.images.shape
        series = torch.zeros((b, t, 17), dtype=ts.images.dtype)
        mask = torch.ones((b, t, 17))

        # svi primjeri u grupi (pa i datasetu) pocinju u isti mjesec
        month = torch.ones((b), dtype=torch.long) * self._month_start

        for m, ts in x.modalities.items():
            # ako je godina postavljena na nulu, nemamo podatke za taj mjesec
            valid_timesteps_mask = (ts.timesteps[..., 0] != 0).unsqueeze(-1)
            if m is Modality.SENTINEL_2_L2A:
                series[..., self._s2_out_channels] = ts.images[
                    ..., self._s2_in_channels
                ]
                mask[..., self._s2_out_channels] = torch.where(
                    valid_timesteps_mask, 0.0, 1.0
                )
            else:
                series[..., self._s1_out_channels] = ts.images
                mask[..., self._s1_out_channels] = torch.where(
                    valid_timesteps_mask, 0.0, 1.0
                )

        return PrestoDatasetSample(
            series=series,
            target=x.target,
            timesteps=ymdh,
            latlon=x.latlon,
            dynamic_world=torch.ones((b, t), dtype=torch.long) * 9,
            mask=mask,
            month=month,
        )
