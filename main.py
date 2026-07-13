import argparse
import time

import lovely_tensors as lt
import numpy as np
import torch

import core.transforms as t
from core.datasets import (
    BalancedBatchSampler,
    MultimodalTimeSeriesDataset,
    DatasetSplit,
    AugmentedDataset,
    multimodal_pad_collate_fn,
    DatasetInstance,
    Modality,
    SingleMonthDataset,
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
        clip=1.0,
        epochs=50,
        batch_size=5,
        batch_size_test=5,
        num_workers=4,
    )

    cfg_optim = {
        "name": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
    }

    cfg_scheduler = {
        "name": "CosineAnnealingLR",
        "T_max": cfg_train.epochs,
    }

    cfg_model = {
        "name": "tae",
        "in_size": 13,
        "hidden_size": 128,
        "embedd_dim": 128,
        "num_heads": 8,
        "out_size": 2,
        "head": [512, 256, 128, 64],
        "dropout": 0.4,
    }

    start_month = 4
    end_month = 7

    DATASET_ROOT = "/mnt/teratron/data/amorfa"
    collate_fn = multimodal_pad_collate_fn
    ds_instance = DatasetInstance.REGIONAL
    curr_modalities = [Modality.SENTINEL_2_L2A]  # , Modality.SENTINEL_1_ASC]
    core_train_ds = MultimodalTimeSeriesDataset(
        DATASET_ROOT,
        ds_instance,
        DatasetSplit.TRAIN,
        curr_modalities,
        # month=start_month,
    )
    data_stats = core_train_ds.get_train_stats(Modality.SENTINEL_2_L2A)

    # class 4 is empty
    class_counts = torch.Tensor(
        data_stats["class_counts"][1:-2]
    )  # TODO: spojit razrede
    # class_counts[0] = class_counts[1]  # poduzorkuj 0 na 1

    class_counts[1] = torch.sum(class_counts[1:])
    class_counts = class_counts[:2]
    class_freq = 1 / class_counts
    class_weights = class_freq / class_freq.norm()
    # class_weights = torch.concat((class_weights, torch.Tensor([0])))
    # class_weights = torch.tensor([0.05, 2])

    cfg_loss = {
        "name": "FocalLoss",
        "weight": class_weights.to(device),
        # "alpha": 0.1,
        # "beta": 1,
        # "num_classes": 3,
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
        positive_ratio=0.375,
        seed=cfg_train.seed,
    )
    batch_sampler = None

    label_mapper = torch.Tensor(
        [
            [1, 0],
            [0, 1],
            [0, 1],
            [0.25, 0.75],
            [0.75, 0.25],
        ]
    )

    train_transforms = t.Compose(
        [
            # t.BiasedRandomCrop(width=100, height=100, seed=cfg_train.seed),
            # t.ApplyToModality(
            #     t.MonthlyRandomSample(
            #         month_start=start_month,
            #         month_end=end_month,
            #         seed=cfg_train.seed,
            #     ),
            #     curr_modalities,
            # ),
            t.ToTensor(),
            t.ApplyToModality(
                t.Compose(
                    [
                        t.Scale(1e-4, dim=1),
                        t.AddNDVI(dim=1, r_idx=3, nir_idx=7),
                        t.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
                    ]
                ),
                Modality.SENTINEL_2_L2A,
            ),
            # t.ApplyToModality(
            #     t.Compose(
            #         [
            #             # t.AddVvVhRatio(dim=1, vv_idx=0, vh_idx=1),
            #             t.Translate(25, dim=1),
            #             t.Scale(1 / 25, dim=1),
            #         ]
            #     ),
            #     Modality.SENTINEL_1_ASC,
            # ),
        ]
    )
    test_transforms = t.Compose(
        [
            # t.ApplyToModality(
            #     t.MonthlyRandomSample(
            #         month_start=start_month,
            #         month_end=end_month,
            #         seed=cfg_train.seed,
            #     ),
            #     curr_modalities,
            # ),
            t.ToTensor(),
            t.ApplyToModality(
                t.Compose(
                    [
                        t.Scale(1e-4, dim=1),
                        t.AddNDVI(dim=1, r_idx=3, nir_idx=7),
                        t.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
                    ]
                ),
                Modality.SENTINEL_2_L2A,
            ),
            # t.ApplyToModality(
            #     t.Compose(
            #         [
            #             # t.AddVvVhRatio(dim=1, vv_idx=0, vh_idx=1),
            #             t.Translate(25, dim=1),
            #             t.Scale(1 / 25, dim=1),
            #         ]
            #     ),
            #     Modality.SENTINEL_1_ASC,
            # ),
        ]
    )
    batch_transforms_train = t.Compose(
        [
            t.BatchSpatialFlatten(batch_first=True),
            t.BatchFilterOut(labels=-1),
            # t.BatchUndersamplingBalancer(reference_cls=1, undersample_cls=0),
            # t.ToPrestoFormat(month_start=start_month),
            t.ConcatenateModalities(curr_modalities, dim=-1),
            t.ToSparseSeries(start_month),
            t.MapLabels(mapper=label_mapper),
        ]
    )
    batch_transforms_test = t.Compose(
        [
            t.BatchSpatialFlatten(batch_first=True),
            t.BatchFilterOut(labels=-1),
            # t.ToPrestoFormat(month_start=start_month),
            t.ConcatenateModalities(curr_modalities, dim=-1),
            t.ToSparseSeries(start_month),
            t.MapLabels(mapper=label_mapper),
        ]
    )

    train_ds = AugmentedDataset(core_train_ds, train_transforms)
    train_eval_ds = AugmentedDataset(
        MultimodalTimeSeriesDataset(
            DATASET_ROOT,
            ds_instance,
            DatasetSplit.TRAIN,
            curr_modalities,
            # start_month,
        ),
        test_transforms,
    )
    val_ds = AugmentedDataset(
        MultimodalTimeSeriesDataset(
            DATASET_ROOT,
            ds_instance,
            DatasetSplit.VALIDATION,
            curr_modalities,
            # start_month,
        ),
        test_transforms,
    )
    test_ds = AugmentedDataset(
        MultimodalTimeSeriesDataset(
            DATASET_ROOT,
            ds_instance,
            DatasetSplit.TEST,
            curr_modalities,
            # start_month,
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
        train_eval_ds,
        val_ds,
        test_ds,
        collate_fn,
        device,
        batch_transforms_train,
        batch_transforms_test,
        batch_sampler,
    )
