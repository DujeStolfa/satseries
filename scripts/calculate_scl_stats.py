import argparse
import os
import csv

import geopandas as gpd
import numpy as np
import rasterio

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Download Sentinel 2 images")
parser.add_argument("root")
parser.add_argument("tiles")
args = parser.parse_args()

scl_valid = [4, 5, 6, 7, 10, 11]
tiles = gpd.read_file(args.tiles)

out_file = os.path.join(args.root, "s2_stats.csv")

with open(out_file, "w", newline="") as outf:
    writer = csv.writer(outf)
    writer.writerow(
        [
            "acquisition_date",
            "valid_percent",
            "series_id",
            "width",
            "height",
            *[f"scl_{i}" for i in range(12)],
        ]
    )

for i, row in tqdm(tiles.iterrows(), desc="Series", position=0):
    csv_rows = []
    series_root = os.path.join(args.root, "series", str(row.series_id), "l2a")

    for f in tqdm(os.listdir(series_root), desc="Images", position=1, leave=False):
        if not f.endswith(".tif"):
            continue

        with rasterio.open(os.path.join(series_root, f)) as ds:
            data = ds.read(13)

        hist, _ = np.histogram(data, bins=np.arange(13), density=True)
        name, _ = os.path.splitext(f)

        csv_rows.append(
            [
                name,
                np.sum(hist[scl_valid]),
                row.series_id,
                ds.width,
                ds.height,
                *hist.tolist(),
            ]
        )

    with open(out_file, "a", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerows(csv_rows)
