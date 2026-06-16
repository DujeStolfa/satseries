import argparse
import time

import lovely_tensors as lt
import numpy as np
import torch

import core.transforms as t
from core.datasets import (
    BalancedBatchSampler,
    SatelliteTimeSeriesDataset,
    DatasetSplit,
    AugmentedDataset,
    pad_collate_fn,
)
from experiment.configs import ExperimentConfig, TrainingConfig
from experiment.run import run_experiment


def ratios_to_posneg_indices(ratios, positive_labels):
    ratios = ratios.T
    positives = np.any(ratios[positive_labels] > 0, axis=0)
    positive_indices = np.argwhere(positives).flatten()
    negative_indices = np.argwhere(~positives).flatten()
    return positive_indices, negative_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("name")
    args = parser.parse_args()

    lt.monkey_patch()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_train = TrainingConfig(
        clip=2.0,
        epochs=50,
        batch_size=16,
        batch_size_test=8,
        num_workers=4,
    )

    cfg_optim = {
        "name": "Adam",
        "lr": 5e-4,
        "weight_decay": 1e-2,
    }

    cfg_scheduler = {
        "name": "CosineAnnealingLR",
        "T_max": cfg_train.epochs,
    }

    cfg_model = {
        "name": "recurrent",
        "in_size": 13,
        "hidden_size": 32,
        "out_size": 5,
        "rnn_cell": "rnn",
        "num_layers": 3,
        "dropout": 0.3,
        "bidirectional": True,
        "attend": False,
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
    class_counts = torch.Tensor(
        data_stats["class_counts"][1:-1]
    )  # TODO: spojit razrede
    class_counts[0] = class_counts[1]  # poduzorkuj 0 na 1

    class_freq = 1 / class_counts
    class_weights = class_freq / class_freq.norm()
    class_weights = torch.concat((class_weights, torch.Tensor([0])))

    cfg_loss = {
        "name": "SymmetricCrossEntropyLoss",
        "weight": class_weights.to(device),
        "alpha": 0.1,
        "beta": 1,
        "num_classes": 5,
        # "gamma": 2,
    }

    pos_indices, neg_indices = ratios_to_posneg_indices(
        ratios=core_train_ds.get_class_ratios(),
        positive_labels=[1, 2, 3, 4],
    )
    batch_sampler = BalancedBatchSampler(
        pos_indices,
        neg_indices,
        batch_size=cfg_train.batch_size,
        positive_ratio=0.5,
        seed=cfg_train.seed,
    )

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
    batch_transforms_train = t.Compose(
        [
            t.BatchSpatialFlatten(batch_first=True),
            t.BatchFilterOut(labels=-1),
            t.BatchUndersamplingBalancer(reference_cls=1, undersample_cls=0),
        ]
    )
    batch_transforms_test = t.Compose(
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
        loss=cfg_loss,
        optimizer=cfg_optim,
        scheduler=cfg_scheduler,
        data={
            "train": train_ds.config_dict,
            "collate_fn": None if collate_fn is None else collate_fn.__name__,
            "instance": ds_instance.value,
        },
        transforms={
            "train": train_transforms.config_dict,
            "testval": test_transforms.config_dict,
            "batch_train": batch_transforms_train.config_dict,
            "batch_testval": batch_transforms_test.config_dict,
        },
        batch_sampler=None if batch_sampler is None else batch_sampler.config_dict,
    )

    run_name = time.strftime(f"%Y%m%d_%H%M", time.localtime())

    run_experiment(
        run_name,
        cfg,
        train_ds,
        val_ds,
        test_ds,
        collate_fn,
        device,
        batch_transforms_train,
        batch_transforms_test,
        batch_sampler,
    )
