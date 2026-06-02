import argparse
import geopandas as gpd
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from acquisition.ingestion import Sentinel2L2ARequest


def download_single_image(request, curr_tile, dt, out_dir, crs, retries):
    timestamp = dt.strftime("%Y-%m-%dT%H-%M-%S")
    outfile = os.path.join(out_dir, f"{timestamp}.tif")

    if os.path.exists(outfile):
        return "exists"

    for attempt in range(retries):
        try:
            data = request.get_data(dt, curr_tile.geometry, crs)

            if data is None:
                return "nodata"

            with open(outfile, "wb") as out_f:
                out_f.write(data)

            return "ok"

        except Exception as e:
            if attempt == retries - 1:
                return f"failed: {e}"

            time.sleep(2**attempt)

    return "failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download time series of Sentinel 2 images"
    )
    parser.add_argument("tiles")
    parser.add_argument("out")
    parser.add_argument("--series", type=int, nargs="+")
    parser.add_argument("--max-cc", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    tiles = gpd.read_file(args.tiles)
    ids = args.series if args.series is not None else tiles.series_id.unique()
    tiles = tiles.set_index("series_id")

    request = Sentinel2L2ARequest()

    for series_id in tqdm(ids, desc="Series", position=0):
        curr_tile = tiles.loc[series_id]

        out_dir = os.path.join(args.out, str(series_id), "l2a")
        os.makedirs(out_dir, exist_ok=True)

        acqs = request.unique_acquisitions(
            curr_tile.start_date,
            curr_tile.end_date,
            curr_tile.geometry,
            tiles.crs,
            max_cloud_cover=args.max_cc,
        )

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    download_single_image,
                    request,
                    curr_tile,
                    dt,
                    out_dir,
                    tiles.crs,
                    args.retries,
                ): dt
                for dt in acqs
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Images",
                position=1,
                leave=False,
            ):
                result = future.result()

                if result not in ("ok", "exists"):
                    tqdm.write(f"{series_id}: {result}")
