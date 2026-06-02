import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import core.transforms as t
from core.datasets import (
    TimeSeriesDatasetSample,
    SatelliteTimeSeriesDataset,
    AugmentedDataset,
)


class Stats:
    def __init__(self):
        self.mean = None
        self.var = None
        self.size = 0

    def update(self, item: TimeSeriesDatasetSample):
        images = item.images.float()

        t, c, h, w = images.shape
        curr_size = t * h * w

        curr_mean = torch.mean(images, dim=(0, 2, 3))
        curr_var = torch.var(images, dim=(0, 2, 3), unbiased=False)

        if self.size == 0:
            self.mean = curr_mean
            self.var = curr_var
            self.size = curr_size
        else:
            old_mean = self.mean
            new_size = self.size + curr_size

            self.mean = (self.size * self.mean + curr_size * curr_mean) / new_size

            self.var = (self.size * self.var + curr_size * curr_var) / new_size + (
                self.size * curr_size * (old_mean - curr_mean) ** 2
            ) / new_size**2

            self.size = new_size

    @property
    def std(self):
        return torch.sqrt(self.var)

    def __repr__(self):
        return f"Stats(\n\tmean={self.mean},\n\tvar={self.var},\n\tstd={self.std}\n\tsize={self.size}\n)"


if __name__ == "__main__":
    import pdb

    parser = argparse.ArgumentParser(description="Calculate dataset statistics")
    parser.add_argument("root")
    parser.add_argument("out")
    args = parser.parse_args()

    ds = AugmentedDataset(SatelliteTimeSeriesDataset(args.root), t.ToTensor())
    stats = Stats()

    for item in tqdm(ds):
        stats.update(item)

    print(stats)

    with open(Path(args.out) / "dataset_stats.json", "w") as f:
        json.dump(
            {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "var": stats.var.tolist(),
                "size": stats.size,
            },
            f,
        )
