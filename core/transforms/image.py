from typing import List

import numpy as np
import torch

from core.transforms.utils import _to_tensor
from core.transforms.base import Transform


class Normalize(Transform):
    def __init__(self, mean, std):
        self._mean = _to_tensor(mean)[:, None, None]
        self._std = _to_tensor(std)[:, None, None]

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["mean"] = self._mean.tolist()
        cfg["std"] = self._std.tolist()
        return cfg

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            return (x - self._mean) / self._std

        x.images = (x.images - self._mean) / self._std
        return x


class Scale(Transform):
    def __init__(self, factor, dim: int, indices: List[int] = None):
        self._factor = factor
        self._dim = dim
        self._indices = indices

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["factor"] = self._factor
        cfg["dim"] = self._dim
        cfg["indices"] = self._indices
        return cfg

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            num_dims = len(x.shape)
        else:
            num_dims = len(x.images.shape)

        curr_slice = [slice(None)] * num_dims
        if self._indices is not None and len(self._indices) != 0:
            curr_slice[self._dim] = self._indices

        if isinstance(x, (torch.Tensor, np.ndarray)):
            x[*curr_slice] = self._factor * x[*curr_slice]
        else:
            x.images[*curr_slice] = self._factor * x.images[*curr_slice]

        return x


class Translate(Transform):
    def __init__(self, shift, dim: int, indices: List[int] = None):
        self._shift = shift
        self._dim = dim
        self._indices = indices

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["shift"] = self._shift
        cfg["dim"] = self._dim
        cfg["indices"] = self._indices
        return cfg

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            num_dims = len(x.shape)
        else:
            num_dims = len(x.images.shape)

        curr_slice = [slice(None)] * num_dims
        if self._indices is not None and len(self._indices) != 0:
            curr_slice[self._dim] = self._indices

        if isinstance(x, (torch.Tensor, np.ndarray)):
            x[*curr_slice] = x[*curr_slice] + self._shift
        else:
            x.images[*curr_slice] = x.images[*curr_slice] + self._shift

        return x


class AddNDVI(Transform):
    def __init__(self, dim, r_idx, nir_idx):
        super().__init__()
        self._dim = dim
        self._r_idx = r_idx
        self._nir_idx = nir_idx

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["dim"] = self._dim
        cfg["r_idx"] = self._r_idx
        cfg["nir_idx"] = self._nir_idx
        return cfg

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            num_dims = len(x.shape)
        else:
            num_dims = len(x.images.shape)

        red_slice = [slice(None)] * num_dims
        nir_slice = [slice(None)] * num_dims
        red_slice[self._dim] = self._r_idx
        nir_slice[self._dim] = self._nir_idx

        if isinstance(x, (torch.Tensor, np.ndarray)):
            red = x[*red_slice]
            nir = x[*nir_slice]
        else:
            red = x.images[*red_slice]
            nir = x.images[*nir_slice]

        ndvi = (nir - red) / (nir + red + 1e-10)

        if isinstance(x, (torch.Tensor, np.ndarray)):
            return torch.cat([x, ndvi.unsqueeze(self._dim)], dim=self._dim)

        x.images = torch.cat([x.images, ndvi.unsqueeze(self._dim)], dim=self._dim)
        return x


class AddVvVhRatio(Transform):
    def __init__(self, dim, vv_idx, vh_idx):
        super().__init__()
        self._dim = dim
        self._vv_idx = vv_idx
        self._vh_idx = vh_idx

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["dim"] = self._dim
        cfg["vv_idx"] = self._vv_idx
        cfg["vh_idx"] = self._vh_idx
        return cfg

    def __call__(self, x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            num_dims = len(x.shape)
        else:
            num_dims = len(x.images.shape)

        vv_slice = [slice(None)] * num_dims
        vh_slice = [slice(None)] * num_dims
        vv_slice[self._dim] = self._vv_idx
        vh_slice[self._dim] = self._vh_idx

        if isinstance(x, (torch.Tensor, np.ndarray)):
            vv = x[*vv_slice]
            vh = x[*vh_slice]
        else:
            vv = x.images[*vv_slice]
            vh = x.images[*vh_slice]

        ratio = vv - vh

        if isinstance(x, (torch.Tensor, np.ndarray)):
            return torch.cat([x, ratio.unsqueeze(self._dim)], dim=self._dim)

        x.images = torch.cat([x.images, ratio.unsqueeze(self._dim)], dim=self._dim)
        return x
