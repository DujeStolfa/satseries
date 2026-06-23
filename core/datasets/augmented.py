import torch.utils.data as data


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
