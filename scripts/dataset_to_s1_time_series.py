import argparse
import os
import json

import pandas as pd
import numpy as np
import rasterio

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def save_series(df, series_id, root, look_direction):
    try:
        curr_series = df.loc[[series_id]].sort_values("acquisition_date")

        image_paths = [
            os.path.join(root, str(series_id), look_direction, f"{date}.tif")
            for date in curr_series["acquisition_date"]
        ]

        out_data = os.path.join(root, str(series_id), f"{look_direction}.npy")
        images = None

        for i, path in enumerate(image_paths):
            with rasterio.open(path) as ds:
                data = ds.read()[:-1]

                if images is None:
                    # Alociraj prazan niz
                    images = np.empty((len(image_paths), *data.shape), dtype=data.dtype)

                images[i] = data

            if os.path.exists(out_data):
                break

        out_meta = os.path.join(root, str(series_id), "metadata.json")
        with open(out_meta, "r") as f:
            metadata = json.load(f)

        metadata[f"dates_{look_direction}"] = curr_series["acquisition_date"].tolist()

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
        description="Group individual Sentinel 1 images into time series cubes"
    )
    parser.add_argument("look_direction")
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
                save_series, selected_images, series_id, args.root, args.look_direction
            ): series_id
            for series_id in ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Series"):
            result = future.result()

            if result != "ok":
                tqdm.write(result)
