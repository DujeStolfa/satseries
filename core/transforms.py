from abc import ABC, abstractmethod
from typing import List

import torch
import numpy as np


def _to_tensor(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.Tensor(x)


class Transform(ABC):

    @property
    def config_dict(self):
        return dict(name=type(self).__name__)

    @abstractmethod
    def __call__(self, data):
        raise NotImplementedError()


class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self._transforms = transforms

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["transforms"] = [t.config_dict for t in self._transforms]
        return cfg

    def __call__(self, x):
        for t in self._transforms:
            x = t(x)
        return x


class ToTensor(Transform):
    def __call__(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()

        x.images = torch.from_numpy(x.images).float()
        x.mask = torch.from_numpy(x.mask)
        x.timesteps = torch.from_numpy(x.timesteps).to(torch.int16)
        x.latlon = torch.from_numpy(x.latlon).float()
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


class BiasedRandomCrop(Transform):
    def __init__(self, width, height, background=-1, seed=None):
        self._crop_width = width
        self._crop_height = height
        self._background = background
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["width"] = self._crop_width
        cfg["height"] = self._crop_height
        cfg["background"] = self._background
        cfg["seed"] = self._seed
        return cfg

    def __call__(self, x):
        mask_data = x.mask.numpy() if hasattr(x.mask, "numpy") else x.mask
        valid_pixels = np.argwhere(mask_data != self._background)

        img_height, img_width = x.mask.shape[-2:]

        if len(valid_pixels) == 0:
            top = self._rng.integers(0, img_height - self._crop_height + 1)
            left = self._rng.integers(0, img_width - self._crop_width + 1)

        else:
            valid_y, valid_x = valid_pixels[self._rng.integers(0, len(valid_pixels))]

            low_y = max(0, valid_y - self._crop_height + 1)
            high_y = min(img_height - self._crop_height, valid_y)

            low_x = max(0, valid_x - self._crop_width + 1)
            high_x = min(img_width - self._crop_width, valid_x)

            top = self._rng.integers(low_y, high_y + 1) if low_y < high_y else low_y
            left = self._rng.integers(low_x, high_x + 1) if low_x < high_x else low_x

        x.images = x.images[
            ..., top : top + self._crop_height, left : left + self._crop_width
        ]
        x.mask = x.mask[
            ..., top : top + self._crop_height, left : left + self._crop_width
        ]

        return x


class BatchSpatialFlatten(Transform):
    def __init__(self, batch_first: bool):
        self._batch_first = batch_first

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["batch_first"] = self._batch_first
        return cfg

    def __call__(self, batch):
        new_images = batch.images  # [B, T, C, H, W]
        new_mask = batch.mask  # [B, H, W]
        new_timesteps = batch.timesteps  # [B, T, 4 (YMDH)]
        new_latlon = batch.latlon  # [B, 2]

        if not self._batch_first:
            new_images = new_images.swapaxes(0, 1)
            new_timesteps = new_timesteps.swapaxes(0, 1)

        shape = new_images.shape
        b, t, c, h, w = shape

        new_images = new_images.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        new_mask = new_mask.reshape(b * h * w)
        new_timesteps = (
            new_timesteps[:, None, None, :, :]
            .expand(-1, h, w, -1, -1)
            .reshape(b * h * w, t, batch.timesteps.shape[-1])
        )
        new_latlon = (
            new_latlon[:, None, None, :]
            .expand(-1, h, w, -1)
            .reshape(b * h * w, batch.latlon.shape[-1])
        )

        if not self._batch_first:
            new_images = new_images.swapaxes(0, 1)
            new_timesteps = new_timesteps.swapaxes(0, 1)

        batch.images = new_images
        batch.mask = new_mask
        batch.timesteps = new_timesteps
        batch.latlon = new_latlon
        return batch


class MonthlyRandomSample(Transform):
    def __init__(self, month_start, month_end, seed=None):
        if month_end < month_start:
            raise ValueError(f"End month can't be before start month")

        if month_start <= 0:
            raise ValueError(f"Invalid month {month_start}")

        if month_end > 12:
            raise ValueError(f"Invalid month {month_end}")

        self._month_start = month_start
        self._month_end = month_end
        self._rng = np.random.default_rng(seed)
        self._seed = seed

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["month_start"] = self._month_start
        cfg["month_end"] = self._month_end
        cfg["seed"] = self._seed
        return cfg

    def __call__(self, x):
        months = x.timesteps[:, 1]

        values, start_idx = np.unique(months, return_index=True)
        end_idx = np.r_[start_idx[1:], len(months)]  # posmak

        # izbaci mjesece izvan zadanog raspona
        mask = (values >= self._month_start) & (values <= self._month_end)

        values = values[mask]
        start_idx = start_idx[mask]
        end_idx = end_idx[mask]

        # odaberi jednu sliku u svakom mjesecu
        sizes = end_idx - start_idx
        offsets = self._rng.integers(0, sizes)
        selected_idx = start_idx + offsets

        # nadopuni nulama u slucaju da nedostaju podaci za neki mjesec
        month_count = self._month_end - self._month_start + 1
        new_images = np.zeros(
            [month_count, *x.images.shape[1:]],
            dtype=x.images.dtype,
        )
        new_timesteps = np.zeros(
            [month_count, *x.timesteps.shape[1:]],
            dtype=x.timesteps.dtype,
        )

        # stvarne vrijednosti
        new_images[values - self._month_start] = x.images[selected_idx, ...]
        new_timesteps[values - self._month_start] = x.timesteps[selected_idx, ...]

        x.images = new_images
        x.timesteps = new_timesteps
        return x


class AddSpectralFeatures(Transform):
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


class Normalize(Transform):
    def __init__(self, mean, std):
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.Tensor(x)

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


class BatchFilterOut(Transform):
    def __init__(self, labels):
        self._labels = _to_tensor(labels if not isinstance(labels, int) else [labels])

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["labels"] = self._labels
        return cfg

    def __call__(self, x):
        if isinstance(x.mask, np.ndarray):
            mask = ~np.isin(x.mask, self._labels.numpy())
        elif isinstance(x.mask, torch.Tensor):
            mask = ~torch.isin(x.mask, self._labels)
        else:
            mask = ~x.mask.isin(self._labels)

        x.images = x.images[mask]
        x.timesteps = x.timesteps[mask]
        x.latlon = x.latlon[mask]
        x.mask = x.mask[mask]
        return x


class BatchUndersamplingBalancer(Transform):
    def __init__(self, reference_cls: int, undersample_cls: List[int]):
        self._reference = reference_cls
        self._undersample = _to_tensor(
            undersample_cls
            if not isinstance(undersample_cls, int)
            else [undersample_cls]
        )

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["reference_cls"] = self._reference
        cfg["undersample_cls"] = self._undersample.tolist()
        return cfg

    def __call__(self, x):
        ref_mask = torch.isin(x.mask, self._reference)
        ref_count = ref_mask.sum().item()

        # dodijeli svakom primjeru nasumicnu vrijednost
        rand = torch.rand(len(x.mask), device=x.mask.device)
        keep = torch.ones(len(x.mask), dtype=torch.bool, device=x.mask.device)

        for cls in self._undersample:
            idx = torch.nonzero(x.mask == cls, as_tuple=True)[0]

            # odaberi `ref_count` primjera iz svakog razreda s najvecom pridruzenom vrijednosti
            if len(idx) > ref_count:
                _, top = torch.topk(rand[idx], ref_count)
                keep[idx] = False
                keep[idx[top]] = True

        x.images = x.images[keep]
        x.timesteps = x.timesteps[keep]
        x.latlon = x.latlon[keep]
        x.mask = x.mask[keep]
        return x


if __name__ == "__main__":
    import numpy as np

    transform = Compose([ToTensor(), Scale(5, [1, 3])])
    data = np.ones((4, 8))
    print(data)
    print(transform(data))
    print(data)
