import numpy as np

from core.transforms.base import Transform


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


class MonthlyComposite(Transform):
    def __init__(self, month_start, month_end, reduction: str):
        if month_end < month_start:
            raise ValueError(f"End month can't be before start month")

        if month_start <= 0:
            raise ValueError(f"Invalid month {month_start}")

        if month_end > 12:
            raise ValueError(f"Invalid month {month_end}")

        self._month_start = month_start
        self._month_end = month_end

        if reduction not in ["mean", "median"]:
            raise ValueError("Supported reduction methods are 'mean' and 'median'")
        self._reduction = reduction

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["month_start"] = self._month_start
        cfg["month_end"] = self._month_end
        cfg["reduction"] = self._reduction
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

        # izracunaj kompozitne slike
        for val, start, end in zip(values, start_idx, end_idx):
            if self._reduction == "median":
                reduced = np.median(x.images[start:end], axis=0)
            elif self._reduction == "mean":
                reduced = np.mean(x.images[start:end], axis=0)
            else:
                raise ValueError("Supported reduction methods are 'mean' and 'median'")

            new_images[val - self._month_start] = reduced

            new_timesteps[:, 0] = x.timesteps[0, 0]  # godina
            new_timesteps[:, 1] = val  # mjesec
            new_timesteps[:, 2] = 15  #

        x.images = new_images
        x.timesteps = new_timesteps
        return x
