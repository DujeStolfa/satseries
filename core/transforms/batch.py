from typing import List

import numpy as np
import torch

from core.transforms.base import Transform
from core.transforms.utils import _to_tensor


class BatchFilterOut(Transform):
    def __init__(self, labels):
        self._labels = _to_tensor(labels if not isinstance(labels, int) else [labels])

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["labels"] = self._labels
        return cfg

    def __call__(self, x):
        if isinstance(x.target, np.ndarray):
            mask = ~np.isin(x.target, self._labels.numpy())
        elif isinstance(x.target, torch.Tensor):
            mask = ~torch.isin(x.target, self._labels)
        else:
            mask = ~x.target.isin(self._labels)

        x.images = x.images[mask]
        x.timesteps = x.timesteps[mask]
        x.latlon = x.latlon[mask]
        x.target = x.target[mask]
        return x


class BatchUndersamplingBalancer(Transform):
    def __init__(self, reference_cls: int, undersample_cls: List[int]):
        self._reference = reference_cls
        self._undersample = _to_tensor(
            undersample_cls
            if not isinstance(undersample_cls, int)
            else [undersample_cls]
        )

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["reference_cls"] = self._reference
        cfg["undersample_cls"] = self._undersample.tolist()
        return cfg

    def __call__(self, x):
        ref_mask = torch.isin(x.target, self._reference)
        ref_count = ref_mask.sum().item()

        if ref_count == 0:
            return x

        # dodijeli svakom primjeru nasumicnu vrijednost
        rand = torch.rand(len(x.target), device=x.target.device)
        keep = torch.ones(len(x.target), dtype=torch.bool, device=x.target.device)

        for cls in self._undersample:
            idx = torch.nonzero(x.target == cls, as_tuple=True)[0]

            # odaberi `ref_count` primjera iz svakog razreda s najvecom pridruzenom vrijednosti
            if len(idx) > ref_count:
                _, top = torch.topk(rand[idx], ref_count)
                keep[idx] = False
                keep[idx[top]] = True

        x.images = x.images[keep]
        x.timesteps = x.timesteps[keep]
        x.latlon = x.latlon[keep]
        x.target = x.target[keep]
        return x


class BatchSpatialFlatten(Transform):
    def __init__(self, batch_first: bool):
        self._batch_first = batch_first

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["batch_first"] = self._batch_first
        return cfg

    def __call__(self, batch):
        new_images = batch.images  # [B, T, C, H, W]
        new_target = batch.target  # [B, H, W]
        new_timesteps = batch.timesteps  # [B, T, 4 (YMDH)]
        new_latlon = batch.latlon  # [B, 2]

        if not self._batch_first:
            new_images = new_images.swapaxes(0, 1)
            new_timesteps = new_timesteps.swapaxes(0, 1)

        shape = new_images.shape
        b, t, c, h, w = shape

        new_images = new_images.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        new_target = new_target.reshape(b * h * w)
        new_timesteps = (
            new_timesteps[:, None, None, :, :]
            .expand(-1, h, w, -1, -1)
            .reshape(b * h * w, t, batch.timesteps.shape[-1])
        )
        new_latlon = (
            new_latlon[:, None, None, :]
            .expand(-1, h, w, -1)
            .reshape(b * h * w, batch.latlon.shape[-1])
        )

        if not self._batch_first:
            new_images = new_images.swapaxes(0, 1)
            new_timesteps = new_timesteps.swapaxes(0, 1)

        batch.images = new_images
        batch.target = new_target
        batch.timesteps = new_timesteps
        batch.latlon = new_latlon
        return batch
