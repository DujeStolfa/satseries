from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.utils as utils

from core.datasets.base import Modality


@dataclass
class UnimodalTimeSeries:
    images: torch.Tensor | np.ndarray
    timesteps: torch.Tensor | np.ndarray

    def to_device(self, device):
        self.images = self.images.to(device)
        self.timesteps = self.timesteps.to(device)
        return self

    def to_tensor(self):
        self.images = torch.from_numpy(self.images)
        self.timesteps = torch.from_numpy(self.timesteps)
        return self


@dataclass
class UnimodalDatasetSample:
    series: torch.Tensor | np.ndarray
    target: torch.Tensor | np.ndarray
    timesteps: torch.Tensor | np.ndarray

    def to_device(self, device):
        self.series = self.series.to(device)
        self.target = self.target.to(device)
        self.timesteps = self.timesteps.to(device)

    def to_tensor(self):
        self.series = torch.from_numpy(self.series)
        self.target = torch.from_numpy(self.target)
        self.timesteps = torch.from_numpy(self.timesteps)
        return self


@dataclass
class MultimodalDatasetSample:
    modalities: Dict[Modality, UnimodalTimeSeries]
    target: torch.Tensor | np.ndarray
    latlon: torch.Tensor | np.ndarray

    def to_device(self, device):
        for ts in self.modalities.values():
            ts.to_device(device)

        self.target = self.target.to(device)
        self.latlon = self.latlon.to(device)
        return self

    def to_tensor(self):
        for ts in self.modalities.values():
            ts.to_tensor()

        self.target = torch.from_numpy(self.target)
        self.latlon = torch.from_numpy(self.latlon)
        return self


@dataclass
class PrestoDatasetSample:
    series: torch.Tensor
    target: torch.Tensor
    timesteps: torch.Tensor
    latlon: torch.Tensor
    dynamic_world: torch.Tensor
    mask: torch.Tensor
    month: torch.Tensor | int

    def to_device(self, device):
        self.series = self.series.to(device)
        self.target = self.target.to(device)
        self.timesteps = self.timesteps.to(device)
        self.latlon = self.latlon.to(device)
        self.dynamic_world = self.dynamic_world.to(device)
        self.mask = self.mask.to(device)
        self.month = (
            self.month.to(device)
            if isinstance(self.month, torch.Tensor)
            else self.month
        )
        return self

    def to_tensor(self):
        return self


def multimodal_pad_collate_fn(
    batch: List[MultimodalDatasetSample], padding_value=0
) -> Tuple[MultimodalDatasetSample, Dict[Modality, torch.Tensor]]:
    latlon = torch.stack([el.latlon for el in batch])
    target = torch.stack([el.target for el in batch])

    data = dict()
    lengths = dict()
    curr_modalities = list(batch[0].modalities.keys())

    for modality in curr_modalities:
        images = [el.modalities[modality].images for el in batch]
        timesteps = [el.modalities[modality].timesteps for el in batch]

        data[modality] = UnimodalTimeSeries(
            images=utils.rnn.pad_sequence(
                images, padding_value=padding_value, batch_first=True
            ),
            timesteps=utils.rnn.pad_sequence(
                timesteps, padding_value=padding_value, batch_first=True
            ),
        )
        lengths[modality] = torch.tensor(
            [len(ts) for ts in timesteps], dtype=torch.uint32
        )

    return (
        MultimodalDatasetSample(
            modalities=data,
            target=target,
            latlon=latlon,
        ),
        lengths,
    )
