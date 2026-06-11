import argparse
import time

import numpy as np
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

    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("name")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_train = TrainingConfig(
        lr=5e-4,
        wd=1e-2,
        clip=2.0,
        epochs=50,
        batch_size=8,
        batch_size_test=1,
    )

    cfg_model = {
        "name": "recurrent",
        "in_size": 13,
        "hidden_size": 32,
        "out_size": 5,
        "rnn_cell": "rnn",
        "num_layers": 3,
        "dropout": 0.3,
        "bidirectional": True,
        "attend": True,
    }

    DATASET_ROOT = "E:\\data\\diplomski\\amorfa"
    collate_fn = pad_collate_fn
    ds_instance = SatelliteTimeSeriesDataset.DatasetInstance.REGIONAL
    core_train_ds = SatelliteTimeSeriesDataset(
        DATASET_ROOT,
        ds_instance,
        DatasetSplit.TRAIN,
    )
    data_stats = core_train_ds.get_train_stats()

    # class 4 is empty
    class_freq = 1 / torch.Tensor(data_stats["class_counts"][1:-1])
    class_weights = class_freq / class_freq.norm()
    class_weights = torch.concat((class_weights, torch.Tensor([0])))
    # class_weights = None

    train_transforms = t.Compose(
        [
            # t.BiasedRandomCrop(width=100, height=100, seed=cfg_train.seed),
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
    batch_transforms = t.Compose(
        [
            t.BatchSpatialFlatten(batch_first=True),
            t.BatchFilterOut(labels=-1),
        ]
    )

    train_ds = AugmentedDataset(core_train_ds, train_transforms)
    val_ds = AugmentedDataset(
        SatelliteTimeSeriesDataset(
            DATASET_ROOT,
            ds_instance,
            DatasetSplit.VALIDATION,
        ),
        test_transforms,
    )
    test_ds = AugmentedDataset(
        SatelliteTimeSeriesDataset(
            DATASET_ROOT,
            ds_instance,
            DatasetSplit.TEST,
        ),
        test_transforms,
    )

    cfg = ExperimentConfig(
        name=args.name,
        training=cfg_train,
        model=cfg_model,
        data={
            "train": train_ds.config_dict,
            "collate_fn": None if collate_fn is None else collate_fn.__name__,
            "instance": ds_instance.value,
            "class_weights": (
                class_weights.tolist() if class_weights is not None else None
            ),
        },
        transforms={
            "train": train_transforms.config_dict,
            "testval": test_transforms.config_dict,
            "batch": batch_transforms.config_dict,
        },
    )
    run_name = time.strftime(f"%Y%m%d_%H%M", time.gmtime())
    run_experiment(
        run_name,
        cfg,
        train_ds,
        val_ds,
        test_ds,
        collate_fn,
        device,
        batch_transforms,
        class_weights,
    )
