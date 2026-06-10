from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    lr: float = 1e-3
    wd: float = 0
    clip: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    batch_size_test: int = 32
    seed: int = 1414213


@dataclass
class ExperimentConfig:
    name: str
    model: dict
    training: TrainingConfig
    transforms: List[dict]
