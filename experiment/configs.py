from dataclasses import dataclass


@dataclass
class TrainingConfig:
    clip: float = 1e-3
    epochs: int = 10
    batch_size: int = 64
    batch_size_test: int = 32
    seed: int = 1414213
    num_workers: int = 0


@dataclass
class ExperimentConfig:
    name: str
    model: dict
    loss: dict
    optimizer: dict
    scheduler: dict
    training: TrainingConfig
    data: dict
    transforms: dict
    batch_sampler: dict
