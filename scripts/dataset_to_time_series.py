import argparse
import os
import json

import pandas as pd
import pyproj
import numpy as np
import rasterio

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def save_series(df, series_id, root):
    try:
        curr_series = df.loc[[series_id]].sort_values("acquisition_date")

        image_paths = [
            os.path.join(root, str(series_id), "l2a", f"{date}.tif")
            for date in curr_series["acquisition_date"]
        ]

        metadata = {
            "series_id": int(series_id),
            "dates_S2": curr_series["acquisition_date"].tolist(),
        }

        out_data = os.path.join(root, str(series_id), "l2a.npy")
        out_meta = os.path.join(root, str(series_id), "metadata.json")

        images = None

        for i, path in enumerate(image_paths):
            with rasterio.open(path) as ds:
                data = ds.read()[:-1]

                if images is None:
                    # Alociraj prazan niz
                    images = np.empty((len(image_paths), *data.shape), dtype=data.dtype)

                    # Ucitaj ostale metapodatke
                    transformer = pyproj.Transformer.from_crs(
                        "EPSG:3765", "EPSG:4326", always_xy=True
                    )

                    x, y = ds.transform * (data.shape[-2] / 2, data.shape[-1] / 2)
                    lon, lat = transformer.transform(x, y)
                    metadata["lon"] = lon
                    metadata["lat"] = lat

                images[i] = data

            if os.path.exists(out_data):
                break

        # if not os.path.exists(out_meta):
        with open(out_meta, "w") as f:
            json.dump(metadata, f)

        if not os.path.exists(out_data):
            with open(out_data, "wb") as f:
                np.save(f, images)

        return "ok"

    except Exception as e:
        return f"{series_id}: {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Group individual Sentinel 2 images into time series cubes"
    )
    parser.add_argument("valid")
    parser.add_argument("root")
    parser.add_argument("--series", type=int, nargs="+")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    selected_images = pd.read_csv(args.valid)
    selected_images = selected_images[selected_images["valid"] != False]

    ids = args.series if args.series is not None else selected_images.series_id.unique()
    selected_images = selected_images.set_index("series_id")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                save_series, selected_images, series_id, args.root
            ): series_id
            for series_id in ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Series"):
            result = future.result()

            if result != "ok":
                tqdm.write(result)
