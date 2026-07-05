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
        self._modalities = modalities if isinstance(modalities, list) else [modalities]

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
            images = np.load(curr_path / f"{modality.value}.npy").astype(np.float32)
            if (
                modality is Modality.SENTINEL_1_ASC
                or modality is Modality.SENTINEL_1_DES
            ):
                images = np.clip(images, -50, 50)

            ymdh = np.array(
                [
                    _timestamp_to_ymdh(d)
                    for d in curr_series_metadata[f"dates_{modality.value}"]
                ],
                dtype=np.long,
            )

            data[modality] = UnimodalTimeSeries(images=images, timesteps=ymdh)

        with rasterio.open(curr_path / "labels_10m.tif") as ds:
            labels = ds.read(1).astype(np.long)

        latlon = np.array(
            [curr_series_metadata["lat"], curr_series_metadata["lon"]], dtype=np.float32
        )

        return MultimodalDatasetSample(
            modalities=data,
            target=labels,
            latlon=latlon,
        )

    def get_train_stats(self, modality):
        if modality is Modality.SENTINEL_2_L2A:
            return self._metadata["stats"][self._instance.value]

        raise NotImplementedError()

    def get_class_ratios(self) -> np.ndarray:
        sid = self._series_ids[0]
        all_items = self._metadata["series"]
        sample = all_items[str(sid)]["class_ratios"]
        out_arr = np.empty((len(self._series_ids), len(sample)))

        for i, sid in enumerate(self._series_ids):
            out_arr[i] = all_items[str(sid)]["class_ratios"]

        return out_arr


class SingleMonthDataset(MultimodalTimeSeriesDataset):
    def __init__(
        self,
        root_dir,
        instance: DatasetInstance,
        split: DatasetSplit,
        modalities: List[Modality],
        month: int,
    ):
        super().__init__(root_dir, instance, split, modalities)
        self._month = month

        # filter series_ids
        keep_indices = list()

        for i, series_id in enumerate(self._series_ids):
            curr = self._metadata["series"][str(series_id)]
            has_data = True

            for modality in self._modalities:
                has_data = has_data and (
                    month
                    in [
                        dt.strptime(ts, "%Y-%m-%dT%H-%M-%S").month
                        for ts in curr[f"dates_{modality.value}"]
                    ]
                )

            if has_data:
                keep_indices.append(i)

        self._series_ids = np.array(self._series_ids)[keep_indices].tolist()

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
            month=self._month,
        )
