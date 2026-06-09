import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

import core.transforms as t
from core.datasets import TimeSeriesDatasetSample


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
    from core.datasets import SatelliteTimeSeriesDataset, AugmentedDataset, DatasetSplit

    parser = argparse.ArgumentParser(description="Calculate dataset statistics")
    parser.add_argument("root")
    parser.add_argument("out")
    parser.add_argument("instance")
    args = parser.parse_args()

    ds = AugmentedDataset(
        SatelliteTimeSeriesDataset(
            args.root,
            SatelliteTimeSeriesDataset.DatasetInstance(args.instance),
            DatasetSplit.TRAIN,
        ),
        t.Compose(
            [
                t.ToTensor(),
                t.Scale(1e-4, dim=1),
                t.AddSpectralFeatures(dim=1, r_idx=3, nir_idx=7),
            ]
        ),
    )
    stats = Stats()

    pbar = tqdm(ds)
    for item in pbar:
        stats.update(item)
        tqdm.write(f"\nmean={stats.mean},\nstd={stats.std}\033[5F")

    print("\n" * 5)
    print(stats)

    with open(Path(args.out) / f"dataset_stats_{args.instance}.json", "w") as f:
        json.dump(
            {
                "mean": stats.mean.tolist(),
                "std": stats.std.tolist(),
                "var": stats.var.tolist(),
                "size": stats.size,
            },
            f,
        )
