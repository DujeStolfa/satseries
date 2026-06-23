import numpy as np
import torch

from core.datasets import PrestoDatasetSample
from core.transforms.base import Transform


class ToTensor(Transform):
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()

        x.images = torch.from_numpy(x.images).float()
        x.target = torch.from_numpy(x.target)
        x.timesteps = torch.from_numpy(x.timesteps).to(torch.long)
        x.latlon = torch.from_numpy(x.latlon).float()
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

    def __call__(self, x):
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

        x.images = x.images[
            ..., top : top + self._crop_height, left : left + self._crop_width
        ]
        x.target = x.target[
            ..., top : top + self._crop_height, left : left + self._crop_width
        ]

        return x


class ToPrestoFormat(Transform):
    def __init__(self):
        self._out_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16]
        self._in_channels = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

    def __call__(self, x) -> PrestoDatasetSample:
        b, t, _ = x.images.shape

        series = torch.zeros((b, t, 17), dtype=x.images.dtype)
        series[..., self._out_channels] = x.images[..., self._in_channels]

        mask = torch.ones((b, t, 17))
        mask[..., self._out_channels] = 0

        return PrestoDatasetSample(
            series=series,
            target=x.target,
            timesteps=x.timesteps,
            latlon=x.latlon,
            dynamic_world=torch.ones((b, t), dtype=torch.long) * 9,
            mask=mask,
            month=x.timesteps[..., 0, 1].to(torch.long),
        )
