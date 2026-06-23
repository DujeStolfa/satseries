import argparse
import os
import csv

import geopandas as gpd
import numpy as np
import rasterio

from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Calculate valid percent of each Sentinel 1 image"
)
parser.add_argument("root")
parser.add_argument("tiles")
args = parser.parse_args()

tiles = gpd.read_file(args.tiles)

for look_direction in ["asc", "des"]:
    print(f"Look direction {look_direction}")
    out_file = os.path.join(args.root, f"s1_{look_direction}_stats.csv")

    with open(out_file, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(
            [
                "acquisition_date",
                "valid_percent",
                "series_id",
                "width",
                "height",
            ]
        )

    for i, row in tqdm(tiles.iterrows(), desc="Series", position=0):
        csv_rows = []
        series_root = os.path.join(
            args.root, "series", str(row.series_id), look_direction
        )

        for f in tqdm(os.listdir(series_root), desc="Images", position=1, leave=False):
            if not f.endswith(".tif"):
                continue

            with rasterio.open(os.path.join(series_root, f)) as ds:
                data = ds.read(3)

            valid_percent = np.mean(data)
            name, _ = os.path.splitext(f)

            csv_rows.append(
                [
                    name,
                    valid_percent,
                    row.series_id,
                    ds.width,
                    ds.height,
                ]
            )

        with open(out_file, "a", newline="") as outf:
            writer = csv.writer(outf)
            writer.writerows(csv_rows)
