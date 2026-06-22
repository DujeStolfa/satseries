import random

import matplotlib.pyplot as plt
import torch.utils.data as data
import lovely_tensors as lt

import core.transforms as t
from core.datasets import (
    SatelliteTimeSeriesDataset,
    AugmentedDataset,
    pad_collate_fn,
    DatasetSplit,
)

lt.monkey_patch()


def true_color(image):
    rgb = image[[3, 2, 1]].numpy().transpose(1, 2, 0)
    return rgb / 1000


ds = AugmentedDataset(
    SatelliteTimeSeriesDataset(
        "E:\\data\\diplomski\\amorfa",
        SatelliteTimeSeriesDataset.DatasetInstance.REGIONAL,
        DatasetSplit.TEST,
    ),
    t.Compose(
        [
            t.BiasedRandomCrop(16, 16),
            t.MonthlyRandomSample(month_start=6, month_end=8),
            t.ToTensor(),
        ]
    ),
)

for _ in range(3):
    idx = random.randint(0, len(ds) - 1)
    curr_item = ds[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(true_color(curr_item.images[0]))
    ax2.imshow(curr_item.target)

    fig.suptitle(f"idx = {idx}, shape = {curr_item.images.shape}")
    plt.show()

loader = data.DataLoader(ds, batch_size=4, shuffle=True, collate_fn=pad_collate_fn)
item, lengths = next(iter(loader))

import pdb

pdb.set_trace()
