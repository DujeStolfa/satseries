import datetime as dt
from enum import Enum
import warnings
import shapely

from abc import ABC, abstractmethod
from typing import Self


from sentinelhub.api.catalog import CatalogSearchIterator, SentinelHubCatalog
from sentinelhub.api.process import SentinelHubRequest
from sentinelhub.constants import CRS, MimeType
from sentinelhub.data_collections import DataCollection
from sentinelhub.download.models import DownloadResponse
from sentinelhub.geometry import BBox
from sentinelhub.time_utils import filter_times

from acquisition.utils import check_bbox_fit


class ProcessingLevel(Enum):
    S1_GA = 1
    S1_GD = 2
    S2_L1C = 3
    S2_L2A = 4
    S2_SCL = 5
    PS_OPT = 6
    PS_DM = 7


class IngestionRequest(ABC):
    def __init__(self):
        with open(self.evalscript_path) as f:
            self.evalscript = f.read()

    @property
    @abstractmethod
    def collection(self) -> DataCollection:
        pass

    @property
    @abstractmethod
    def evalscript_path(self) -> str:
        pass

    @property
    @abstractmethod
    def resolution(self) -> float:
        pass

    @property
    @abstractmethod
    def processing_level(self) -> ProcessingLevel:
        pass

    @abstractmethod
    def data_mask_request(self) -> Self | None:
        pass

    def get_data(self, timestamp, bbox, crs):
        """Timestamp mora bit valjan"""
        bbox = _parse_bbox(bbox, crs, self.resolution, self.resolution)
        time_difference = dt.timedelta(hours=1)

        request = self._construct_request(bbox, timestamp, time_difference)
        response: DownloadResponse = request.get_data(decode_data=False)[0]
        return response.content

    def unique_acquisitions(self, dt_from, dt_to, bbox, crs, max_cloud_cover=None):
        bbox = _parse_bbox(bbox, crs, self.resolution, self.resolution)
        filter = (
            f"eo:cloud_cover < {max_cloud_cover}" if max_cloud_cover is not None else ""
        )
        search_iterator = self._search_catalog(dt_from, dt_to, bbox, filter)

        time_difference = dt.timedelta(hours=1)
        all_timestamps = search_iterator.get_timestamps()
        unique_acquisitions = filter_times(all_timestamps, time_difference)

        return unique_acquisitions

    def _construct_request(
        self, bbox, timestamp, time_difference
    ) -> SentinelHubRequest:
        return SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.collection,
                    time_interval=(
                        timestamp - time_difference,
                        timestamp + time_difference,
                    ),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            resolution=(self.resolution, self.resolution),
        )

    def _search_catalog(self, dt_from, dt_to, bbox, filter) -> CatalogSearchIterator:
        catalog = SentinelHubCatalog()

        return catalog.search(
            self.collection,
            bbox=bbox,
            time=(dt_from, dt_to),
            filter=filter,
            fields={
                "include": ["id", "properties.datetime"],
                "exclude": [],
            },
        )


class Sentinel1RequestHooks(IngestionRequest):

    @property
    def resolution(self) -> float:
        return 10

    def data_mask_request(self):
        return None

    def _construct_request(self, bbox, timestamp, time_difference):
        return SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.collection,
                    time_interval=(
                        timestamp - time_difference,
                        timestamp + time_difference,
                    ),
                    other_args={
                        "processing": {
                            "backCoeff": "GAMMA0_TERRAIN",
                            "demInstance": "COPERNICUS_30",
                            "orthorectify": True,
                            "speckleFilter": {
                                "type": "LEE",
                                "windowSizeX": 3,
                                "windowSizeY": 3,
                            },
                        }
                    },
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            resolution=(self.resolution, self.resolution),
        )


class Sentinel1AscRequest(Sentinel1RequestHooks):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.SENTINEL1_IW_ASC

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_s1_asc.js"

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.S1_GA


class Sentinel1DescRequest(Sentinel1RequestHooks):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.SENTINEL1_IW_DES

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_s1_desc.js"

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.S1_GD


class Sentinel2L1CRequest(IngestionRequest):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.SENTINEL2_L1C

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_s2_l1c.js"

    @property
    def resolution(self) -> float:
        return 10

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.S2_L1C


class Sentinel2L2ARequest(IngestionRequest):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.SENTINEL2_L2A

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_s2_l2a.js"

    @property
    def resolution(self) -> float:
        return 10

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.S2_L2A

    def data_mask_request(self):
        return Sentinel2SCLRequest()


class Sentinel2SCLRequest(IngestionRequest):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.SENTINEL2_L2A

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_s2_scl.js"

    @property
    def resolution(self) -> float:
        return 10

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.S2_SCL

    def data_mask_request(self):
        return None


class PlanetScopeOpticalRequest(IngestionRequest):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.define_byoc("21b0a309-f4b5-4553-a255-77adc06f8002")

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_ps_opt.js"

    @property
    def resolution(self) -> float:
        return 3

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.PS_OPT

    def data_mask_request(self):
        return PlanetScopeDataMaskRequest()


class PlanetScopeDataMaskRequest(IngestionRequest):

    @property
    def collection(self) -> DataCollection:
        return DataCollection.define_byoc("21b0a309-f4b5-4553-a255-77adc06f8002")

    @property
    def evalscript_path(self) -> str:
        return "acquisition\\evalscripts\\eval_ps_dm.js"

    @property
    def resolution(self) -> float:
        return 3

    @property
    def processing_level(self) -> ProcessingLevel:
        return ProcessingLevel.PS_DM

    def data_mask_request(self):
        return None


def _parse_bbox(bbox, crs, resolution_x, resolution_y):
    if isinstance(bbox, shapely.Geometry):
        bbox = BBox(shapely.bounds(bbox).tolist(), CRS(crs))

    elif not isinstance(bbox, BBox):
        bbox = BBox(bbox, CRS(crs))

    if not check_bbox_fit(bbox.geometry, crs, resolution_x, resolution_y):
        warnings.warn(
            f"Bounding box dimensions are not multiples of resolution ({resolution_x}, {resolution_y}), imagery will be resampled to fit to given bounding box.",
            category=RuntimeWarning,
        )

    return bbox


if __name__ == "__main__":
    from acquisition.utils import reproject_polygon, expand_bbox_to_resolution
    from rasterio.io import MemoryFile

    polygon = reproject_polygon(
        shapely.geometry.box(
            16.43342831,
            43.32676437,
            16.44346314,
            43.33396576,
        ),
        "EPSG:4326",
        "EPSG:3765",
    )

    # request = Sentinel2SCLRequest()
    request = Sentinel2L2ARequest()
    # request = PlanetScopeOpticalRequest()
    # request = PlanetScopeDataMaskRequest()

    bbox_exp = expand_bbox_to_resolution(
        polygon, "EPSG:3765", request.resolution, request.resolution
    )

    acqs = request.unique_acquisitions(
        dt.datetime(2024, 7, 13),
        dt.datetime(2024, 7, 17),
        bbox_exp,
        "EPSG:3765",
    )

    data, actual_ts = request.get_data(acqs[0], bbox_exp, "EPSG:3765")
    with MemoryFile(data) as f:
        with f.open() as ds:
            print(ds.shape, ds.count)

    if data is not None:
        with open("test.tif", "wb") as out_f:
            out_f.write(data)
        print("Saved to test.tif")
    else:
        print("No data, nothing saved")
