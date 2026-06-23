from abc import ABC, abstractmethod
from typing import List


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
