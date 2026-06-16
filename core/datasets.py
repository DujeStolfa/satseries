import json
import math
from enum import StrEnum
from typing import List

import numpy as np
import rasterio
import torch
import torch.utils.data as data
import torch.nn.utils as utils

from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


@dataclass
class TimeSeriesDatasetSample:
    images: torch.Tensor | np.ndarray
    mask: torch.Tensor | np.ndarray
    timesteps: torch.Tensor | np.ndarray
    latlon: torch.Tensor | np.ndarray


class AugmentedDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index) -> TimeSeriesDatasetSample:
        return self._transform(self._dataset[index])

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            dataset=self._dataset.config_dict,
        )


class SatelliteTimeSeriesDataset(data.Dataset):
    class DatasetInstance(StrEnum):
        REGIONAL = "regional"
        RANDOM = "random"

    def __init__(self, root_dir, instance: DatasetInstance, split: DatasetSplit):
        self._root_dir = Path(root_dir)
        self._instance = instance
        self._split = split

        with open(self._root_dir / "metadata.json") as f:
            self._metadata = json.load(f)

        self._series_ids = self._metadata["splits"][instance.value][split.value]

        series_dir = self._root_dir / "series"
        self._series_dirs = [series_dir / str(sid) for sid in self._series_ids]
        self._series_dirs = [dir for dir in self._series_dirs if dir.is_dir()]

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            root_dir=self._root_dir,
            instance=self._instance.value,
            split=self._split.value,
        )

    def __len__(self):
        return len(self._series_dirs)

    def __getitem__(self, index) -> TimeSeriesDatasetSample:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        curr_path = self._series_dirs[index]
        series_id = str(curr_path.name)

        images = np.load(curr_path / "l2a.npy")
        with rasterio.open(curr_path / "labels_10m.tif") as ds:
            labels = ds.read(1)

        curr_series_metadata = self._metadata["series"][series_id]

        dates = [
            dt.strptime(d, "%Y-%m-%dT%H-%M-%S")
            for d in curr_series_metadata["dates_S2"]
        ]
        ymd = np.array(
            [[d.year, d.month, d.day, d.hour] for d in dates], dtype=np.uint16
        )
        latlon = np.array(
            [curr_series_metadata["lat"], curr_series_metadata["lon"]], dtype=np.float32
        )

        return TimeSeriesDatasetSample(
            images=images,
            mask=labels,
            timesteps=ymd,
            latlon=latlon,
        )

    def get_train_stats(self):
        return self._metadata["stats"][self._instance.value]

    def get_class_ratios(self) -> np.ndarray:
        sid = self._series_ids[0]
        all_items = self._metadata["series"]
        sample = all_items[str(sid)]["class_ratios"]
        out_arr = np.empty((len(self._series_ids), len(sample)))

        for i, sid in enumerate(self._series_ids):
            out_arr[i] = all_items[str(sid)]["class_ratios"]

        return out_arr


def pad_collate_fn(batch: List[TimeSeriesDatasetSample], pad_index=0):
    latlon = torch.stack([el.latlon for el in batch])
    mask = torch.stack([el.mask for el in batch])
    timesteps = [el.timesteps for el in batch]

    padded_images = utils.rnn.pad_sequence(
        [el.images for el in batch], padding_value=pad_index, batch_first=True
    )
    padded_timesteps = utils.rnn.pad_sequence(
        timesteps, padding_value=pad_index, batch_first=True
    )

    lengths = torch.tensor([len(ts) for ts in timesteps], dtype=torch.uint32)

    return (
        TimeSeriesDatasetSample(
            images=padded_images,
            mask=mask,
            timesteps=padded_timesteps,
            latlon=latlon,
        ),
        lengths,
    )


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


if __name__ == "__main__":
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    import core.transforms as t

    def true_color(image):
        rgb = image[[3, 2, 1]].transpose(1, 2, 0)
        return rgb / 1000

    ds = AugmentedDataset(
        SatelliteTimeSeriesDataset(
            "E:\\data\\diplomski\\amorfa",
            SatelliteTimeSeriesDataset.DatasetInstance.REGIONAL,
            DatasetSplit.TEST,
        ),
        t.ToTensor(),
    )
    dataloader = data.DataLoader(ds, 4, shuffle=True, collate_fn=pad_collate_fn)

    item = next(iter(dataloader))

    import pdb

    pdb.set_trace()

    item = ds[100]
    images = item.images.numpy()

    fig = plt.figure()

    i = 0
    im = plt.imshow(true_color(images[i]), animated=True)
    t = plt.title(str(i))

    def updatefig(*args):
        global i
        i = (i + 1) % images.shape[0]
        t.set_text(str(i))
        im.set_array(true_color(images[i]))
        return (im,)

    ani = animation.FuncAnimation(fig, updatefig, interval=1000, blit=True)
    plt.show()
