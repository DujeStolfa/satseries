import argparse
import os
import warnings

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
import rasterio.transform

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Generate segmentation maps from polygons")
parser.add_argument("labels")
parser.add_argument("tiles")
parser.add_argument("out")
parser.add_argument(
    "-ts", "--tile-size", required=True, type=float, help="tile width in meters"
)
parser.add_argument("-s", "--series", type=int, nargs="+"),
args = parser.parse_args()


labels = gpd.read_file(args.labels).set_index("series_id")
tiles = gpd.read_file(args.tiles)

label_map = {
    "Ostalo": 0,
    "Amorfa": 1,
    "Mlada": 2,
    "Miks 1": 3,
    "Miks 2": 4,
}
labels["class_value"] = labels["Klasa"].apply(lambda x: label_map[x])

ids = args.series if args.series is not None else tiles.series_id.unique()
tiles = tiles.set_index("series_id")

for series_id in tqdm(ids):
    curr_labels = labels.loc[[series_id]]

    curr_tile = tiles.loc[series_id].geometry
    west, south, east, north = curr_tile.bounds

    out_dir = os.path.join(args.out, str(series_id))
    os.makedirs(out_dir, exist_ok=True)

    for resolution in [3, 10]:
        width = int((east - west) / resolution)
        height = int((north - south) / resolution)
        transform = rasterio.transform.from_bounds(
            west, south, east, north, width, height
        )

        if (
            width != args.tile_size / resolution
            or height != args.tile_size / resolution
        ):
            warnings.warn(
                f"Unexpected shape ({width}, {height}) for series {series_id} and resolution {resolution}"
            )

        geom_value = curr_labels[["geometry", "class_value"]].itertuples(index=False)

        raster = rasterio.features.rasterize(
            geom_value,
            out_shape=(width, height),
            transform=transform,
            all_touched=False,
            fill=-1,
            dtype=np.int8,
        )

        with rasterio.open(
            os.path.join(out_dir, f"labels_{resolution}m.tif"),
            "w",
            driver="GTiff",
            crs=labels.crs,
            transform=transform,
            dtype=rasterio.int8,
            count=1,
            width=width,
            height=height,
        ) as dst:
            dst.write(raster, indexes=1)
