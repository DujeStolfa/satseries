import json
from typing import List

import numpy as np
import rasterio
import torch
import torch.utils.data as data
import torch.nn.utils as utils

from dataclasses import dataclass
from datetime import datetime as dt
from pathlib import Path


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


class SatelliteTimeSeriesDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)

        # mozda i ovo procitat iz metapodataka?
        series_dir = self.root_dir / "series"
        self.series_dirs = [p for p in series_dir.iterdir() if p.is_dir()]

        with open(self.root_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        # procitat splitove

    def __len__(self):
        return len(self.series_dirs)

    def __getitem__(self, index) -> TimeSeriesDatasetSample:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        curr_path = self.series_dirs[index]
        series_id = str(curr_path.name)

        images = np.load(curr_path / "l2a.npy")
        with rasterio.open(curr_path / "labels_10m.tif") as ds:
            labels = ds.read(1)

        curr_series_metadata = self.metadata["series"][series_id]

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


# def pad_collate_fn(batch, pad_index=0):
#     others, labels = zip(*batch)
#     # images, ymd, latlon = zip(*others)
#     images = others

#     # lengths = torch.tensor([len(date) for date in ymd], dtype=torch.uint32)
#     padded_images = utils.rnn.pad_sequence(
#         images, padding_value=pad_index, batch_first=True
#     )
#     # padded_ymd = utils.rnn.pad_sequence(ymd, padding_value=pad_index, batch_first=True)
#     return padded_images, torch.stack(labels)

#     return (
#         (padded_images, padded_ymd, torch.stack(latlon)),
#         torch.stack(labels),
#         lengths,
#     )


if __name__ == "__main__":
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    import core.transforms as t

    def true_color(image):
        rgb = image[[3, 2, 1]].transpose(1, 2, 0)
        return rgb / 1000

    ds = AugmentedDataset(
        SatelliteTimeSeriesDataset("E:\\data\\diplomski\\amorfa"), t.ToTensor()
    )
    dataloader = data.DataLoader(ds, 4, shuffle=True, collate_fn=pad_collate_fn)

    item = next(iter(dataloader))

    import pdb

    pdb.set_trace()

    item = ds[100]
    images = item.images

    print(ds.series_dirs[100])

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
