from datetime import datetime as dt

import torch.utils.data as data

from core.datasets.multimodal import MultimodalTimeSeriesDataset


class AugmentedDataset(data.Dataset):
    def __init__(self, dataset: data.Dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._transform(self._dataset[index])

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            dataset=self._dataset.config_dict,
        )


class FilteredDataset(data.Datset):
    def __init__(self, dataset: data.Dataset, filtered_indices: list):
        self._dataset = dataset
        self._filter = filtered_indices
        self._index_map = sorted(
            list(set(range(len(self._dataset))) - set(self._filter))
        )

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, index):
        return self._dataset[self._index_map[index]]

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            dataset=self._dataset.config_dict,
            filtered_indices=self._filter,
        )


class SingleMonthDataset(FilteredDataset):
    def __init__(self, dataset: MultimodalTimeSeriesDataset, month: int):
        filtered_indices = list()

        for i, series_id in enumerate(dataset._series_ids):
            curr = dataset._metadata["series"][series_id]
            has_data = True

            for modality in dataset._modalities:
                has_data = has_data and (
                    month
                    in [
                        dt.strptime(ts, "%Y-%m-%dT%H-%M-%S").month
                        for ts in curr[f"dates_{modality.value}"]
                    ]
                )

            if not has_data:
                filtered_indices.append(i)

        super().__init__(dataset, filtered_indices)
        self._month = month

    @property
    def config_dict(self):
        return dict(
            name=type(self).__name__,
            dataset=self._dataset.config_dict,
            month=self._month,
        )
