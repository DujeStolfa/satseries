import random

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import lovely_tensors as lt

from core.datasets.base import Modality
import core.transforms as t
from core.datasets import (
    MultimodalTimeSeriesDataset,
    AugmentedDataset,
    multimodal_pad_collate_fn,
    DatasetSplit,
    DatasetInstance,
)

lt.monkey_patch()


def true_color(image):
    rgb = image[[3, 2, 1]].numpy().transpose(1, 2, 0)
    return rgb / 1000


def sar_composite(image):
    rg = (image - image.min()) / (image.max() - image.min())
    b = image[0] - image[1]
    b = (b - b.min()) / (b.max() - b.min())
    return torch.cat([rg, b.unsqueeze(0)], dim=0).numpy().transpose(1, 2, 0)


ds = AugmentedDataset(
    MultimodalTimeSeriesDataset(
        "E:\\data\\diplomski\\amorfa",
        DatasetInstance.REGIONAL,
        DatasetSplit.TEST,
        [Modality.SENTINEL_2_L2A, Modality.SENTINEL_1_ASC],
    ),
    t.Compose(
        [
            t.BiasedRandomCrop(60, 60),
            t.ApplyToModality(
                t.MonthlyRandomSample(month_start=6, month_end=8),
                Modality.SENTINEL_2_L2A,
            ),
            t.ToTensor(),
        ]
    ),
)

for _ in range(5):
    idx = random.randint(0, len(ds) - 1)
    curr = ds[idx]
    curr_s2 = curr.modalities[Modality.SENTINEL_2_L2A]
    curr_s1 = curr.modalities[Modality.SENTINEL_1_ASC]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(true_color(curr_s2.images[0]))
    ax2.imshow(sar_composite(curr_s1.images[0]))
    ax3.imshow(curr.target)

    fig.suptitle(f"idx = {idx},\ns2 = {curr_s2.images},\ns1 = {curr_s1.images}")
    plt.show()


loader = data.DataLoader(
    ds, batch_size=4, shuffle=True, collate_fn=multimodal_pad_collate_fn
)
item, lengths = next(iter(loader))

import pdb

pdb.set_trace()
