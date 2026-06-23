import torch

from core.transforms.base import Transform


class MapLabels(Transform):
    def __init__(self, mapper):
        self._mapper = mapper

    @property
    def config_dict(self):
        cfg = super().config_dict
        cfg["mapper"] = self._mapper.tolist()
        return cfg

    def __call__(self, x):
        x.target = self._mapper[x.target.to(torch.long)]
        return x
