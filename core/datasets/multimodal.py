import json
from datetime import datetime as dt
from pathlib import Path
from typing import List

import numpy as np
import rasterio
import torch.utils.data as data

from core.datasets.base import DatasetInstance, DatasetSplit, Modality
from core.datasets.types import MultimodalDatasetSample, UnimodalTimeSeries


class MultimodalTimeSeriesDataset(data.Dataset):
    def __init__(
        self,
        root_dir,
        instance: DatasetInstance,
        split: DatasetSplit,
        modalities: List[Modality],
    ):
        self._root_dir = Path(root_dir)
        self._instance = instance
        self._split = split
        self._modalities = modalities

        with open(self._root_dir / "metadata.json") as f:
            self._metadata = json.load(f)

        self._series_ids = self._metadata["splits"][instance.value][split.value]

        series_dir = self._root_dir / "series"
        self._series_dirs = [series_dir / str(sid) for sid in self._series_ids]
        self._series_dirs = [dir for dir in self._series_dirs if dir.is_dir()]

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            root_dir=self._root_dir,
            instance=self._instance.value,
            split=self._split.value,
            modalities=[m.value for m in self._modalities],
        )

    def __len__(self):
        return len(self._series_dirs)

    def __getitem__(self, index) -> MultimodalDatasetSample:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        curr_path = self._series_dirs[index]
        series_id = str(curr_path.name)
        curr_series_metadata = self._metadata["series"][series_id]

        def _timestamp_to_ymdh(timestamp):
            date = dt.strptime(timestamp, "%Y-%m-%dT%H-%M-%S")
            return [date.year, date.month, date.day, date.hour]

        data = dict()
        for modality in self._modalities:
            images = np.load(curr_path / f"{modality.value}.npy")
            ymdh = np.array(
                [
                    _timestamp_to_ymdh(d)
                    for d in curr_series_metadata[f"dates_{modality.value}"]
                ],
                dtype=np.long,
            )

            data[modality] = UnimodalTimeSeries(images=images, timesteps=ymdh)

        with rasterio.open(curr_path / "labels_10m.tif") as ds:
            labels = ds.read(1).to(np.long)

        latlon = np.array(
            [curr_series_metadata["lat"], curr_series_metadata["lon"]], dtype=np.float32
        )

        return MultimodalDatasetSample(
            modalities=data,
            target=labels,
            latlon=latlon,
        )
