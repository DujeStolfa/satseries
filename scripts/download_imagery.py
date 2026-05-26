import argparse
import os
import geopandas as gpd

from tqdm import tqdm
from acquisition.ingestion import Sentinel2L2ARequest

parser = argparse.ArgumentParser(description="Generate segmentation maps from polygons")
parser.add_argument("tiles")
parser.add_argument("out")
parser.add_argument("-s", "--series", type=int, nargs="+")
parser.add_argument("-cc", "--max-cc", type=int)
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

    for dt in tqdm(acqs, desc="Images", position=1, leave=False):
        timestamp = dt.strftime("%Y-%m-%dT%H-%M-%S")
        outfile = os.path.join(out_dir, f"{timestamp}.tif")
        if os.path.exists(outfile):
            continue

        try:
            data = request.get_data(dt, curr_tile.geometry, tiles.crs)
        except Exception as e:
            tqdm.write(f"Failed for {series_id} at {timestamp}: {e}")
            continue

        if data is not None:
            with open(outfile, "wb") as out_f:
                out_f.write(data)
        else:
            tqdm.write(f"No data for series {series_id}")
