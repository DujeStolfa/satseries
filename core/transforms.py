import torch
import numpy as np

from typing import Callable, List


class Compose:
    def __init__(self, transforms: List[Callable]):
        self._transforms = transforms

    def __call__(self, x):
        for t in self._transforms:
            x = t(x)
        return x


class Normalize:
    def __init__(self, mean, std):

        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.Tensor(x)

        self._mean = _to_tensor(mean)
        self._std = _to_tensor(std)

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return (x - self._mean) / self._std

        x.images = (x.images - self._mean) / self._std
        return x


class ToTensor:
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()

        x.images = torch.from_numpy(x.images).float()
        x.mask = torch.from_numpy(x.mask)
        x.timesteps = torch.from_numpy(x.timesteps)
        x.latlon = torch.from_numpy(x.latlon).float()
        return x


class Scale:
    def __init__(self, factor, channels: List[int]):
        self._factor = factor
        self._channels = channels

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            x[..., self._channels] = self._factor * x[..., self._channels]
        else:
            x.images[..., self._channels] = self._factor * x.images[..., self._channels]

        return x


if __name__ == "__main__":
    import numpy as np

    transform = Compose([ToTensor(), Scale(5, [1, 3])])
    data = np.ones((4, 8))
    print(data)
    print(transform(data))
    print(data)
