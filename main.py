import torch

import core.transforms as t
from core.datasets import (
    SatelliteTimeSeriesDataset,
    DatasetSplit,
    AugmentedDataset,
    pad_collate_fn,
)
from experiment.configs import ExperimentConfig, TrainingConfig
from experiment.run import run_experiment

if __name__ == "__main__":
    import lovely_tensors as lt

    lt.monkey_patch()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_train = TrainingConfig(
        lr=2e-3,
        wd=1e-2,
        clip=2.0,
        epochs=2,
        batch_size=16,
        batch_size_test=1,
    )

    cfg_model = {
        "name": "recurrent",
        "in_size": 13,
        "hidden_size": 32,
        "out_size": 5,
        "rnn_cell": "rnn",
        "num_layers": 2,
        "dropout": 0.3,
        "bidirectional": True,
        "attend": True,
    }

    DATASET_ROOT = "E:\\data\\diplomski\\amorfa"
    core_train_ds = SatelliteTimeSeriesDataset(
        DATASET_ROOT,
        SatelliteTimeSeriesDataset.DatasetInstance.REGIONAL,
        DatasetSplit.TRAIN,
    )
    data_stats = core_train_ds.get_train_stats()

    train_transforms = t.Compose(
        [
            t.BiasedRandomCrop(width=8, height=8, seed=cfg_train.seed),
            t.MonthlyRandomSample(
                month_start=4,
                month_end=10,
                seed=cfg_train.seed,
            ),
            t.ToTensor(),
            t.Scale(1e-4, dim=1),
            t.AddSpectralFeatures(dim=1, r_idx=3, nir_idx=7),
            t.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
        ]
    )
    test_transforms = t.Compose(
        [
            t.MonthlyRandomSample(
                month_start=4,
                month_end=10,
                seed=cfg_train.seed,
            ),
            t.ToTensor(),
            t.Scale(1e-4, dim=1),
            t.AddSpectralFeatures(dim=1, r_idx=3, nir_idx=7),
            t.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
        ]
    )
    batch_transforms = t.BatchSpatialFlatten(batch_first=True)

    train_ds = AugmentedDataset(core_train_ds, train_transforms)
    val_ds = AugmentedDataset(
        SatelliteTimeSeriesDataset(
            DATASET_ROOT,
            SatelliteTimeSeriesDataset.DatasetInstance.REGIONAL,
            DatasetSplit.VALIDATION,
        ),
        test_transforms,
    )
    test_ds = AugmentedDataset(
        SatelliteTimeSeriesDataset(
            DATASET_ROOT,
            SatelliteTimeSeriesDataset.DatasetInstance.REGIONAL,
            DatasetSplit.TEST,
        ),
        test_transforms,
    )

    cfg = ExperimentConfig(
        name="Initial experiment",
        training=cfg_train,
        model=cfg_model,
        transforms=[],  # TODO: transforms
    )

    run_experiment(
        cfg, train_ds, val_ds, test_ds, pad_collate_fn, device, batch_transforms
    )
