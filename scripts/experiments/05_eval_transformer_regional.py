import argparse
import csv

import lovely_tensors as lt
import numpy as np
import torch
from mlflow.pytorch import load_model

from core.loops import evaluate
import core.transforms as t
from core.datasets import (
    MultimodalTimeSeriesDataset,
    DatasetSplit,
    AugmentedDataset,
    multimodal_pad_collate_fn,
    DatasetInstance,
    Modality,
)
from core.losses import build_loss
from experiment.configs import TrainingConfig
from experiment.run import _pad_int, load_datasets, set_seed


def ratios_to_posneg_indices(ratios, positive_labels):
    ratios = ratios.T
    positives = np.any(ratios[positive_labels] > 0, axis=0)
    positive_indices = np.argwhere(positives).flatten()
    negative_indices = np.argwhere(~positives).flatten()
    return positive_indices, negative_indices


def eval_csv_rowpart(loss, acc, f1, precision, recall, ap):
    metrics = [loss, acc, f1.mean(), f1.min()]
    for i in range(len(f1)):
        metrics += [
            f1[i],
            precision[i],
            recall[i],
            ap[i],
        ]
    return metrics


def eval_csv_headerpart(split, num_classes):
    header = [
        f"{split}_loss",
        f"{split}_acc",
        f"{split}_f1_macro",
        f"{split}_f1_worst",
    ]
    for i in range(num_classes):
        header += [
            f"{split}_f1_class_{i}",
            f"{split}_precision_class_{i}",
            f"{split}_recall_class_{i}",
            f"{split}_ap_class_{i}",
        ]
    return header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a batch of checkpoints")
    parser.add_argument("in_file")
    parser.add_argument("out_file")
    args = parser.parse_args()

    lt.monkey_patch()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = 2
    start_month = 4
    end_month = 7

    cfg_eval = TrainingConfig(
        clip=1.0,
        epochs=None,
        batch_size=5,
        batch_size_test=5,
        num_workers=4,
    )

    with open(args.out_file, "w", newline="") as f:
        header = ["run_name", "run_id", "epoch"]
        for split in ["train", "val", "test"]:
            header += eval_csv_headerpart(split, num_classes)

        writer = csv.writer(f)
        writer.writerow(header)

    with open(args.in_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        in_rows = [r for r in reader]

    for row in in_rows:
        run_name = row["run_name"]
        run_id = row["run_id"]
        epoch = row["step"]
        ds_instance = DatasetInstance.REGIONAL

        print("- " * 30)
        print(f"Evaulating epoch {epoch} of run {run_id}")
        print("- " * 30)
        set_seed(cfg_eval.seed)

        model = load_model(f"runs:/{run_id}/checkpoint_{_pad_int(epoch)}")
        model.to(device)

        DATASET_ROOT = "/mnt/teratron/data/amorfa"
        collate_fn = multimodal_pad_collate_fn
        curr_modalities = [Modality.SENTINEL_2_L2A]
        core_train_ds = MultimodalTimeSeriesDataset(
            DATASET_ROOT,
            ds_instance,
            DatasetSplit.TRAIN,
            curr_modalities,
        )
        data_stats = core_train_ds.get_train_stats(Modality.SENTINEL_2_L2A)

        # class 4 is empty
        class_counts = torch.Tensor(data_stats["class_counts"][1:-2])
        class_counts[1] = torch.sum(class_counts[1:])
        class_counts = class_counts[:2]
        class_freq = 1 / class_counts
        class_weights = class_freq / class_freq.norm()

        cfg_loss = {
            "name": "FocalLoss",
            "weight": class_weights.to(device),
            "gamma": 2,
        }

        label_mapper = torch.Tensor(
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [0.25, 0.75],
                [0.75, 0.25],
            ]
        )
        batch_sampler = None

        test_transforms = t.Compose(
            [
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
            ]
        )
        batch_transforms_test = t.Compose(
            [
                t.BatchSpatialFlatten(batch_first=True),
                t.BatchFilterOut(labels=-1),
                t.ConcatenateModalities(curr_modalities, dim=-1),
                t.ToSparseSeries(start_month),
                t.MapLabels(mapper=label_mapper),
            ]
        )

        train_eval_ds = AugmentedDataset(
            MultimodalTimeSeriesDataset(
                DATASET_ROOT,
                ds_instance,
                DatasetSplit.TRAIN,
                curr_modalities,
            ),
            test_transforms,
        )
        val_ds = AugmentedDataset(
            MultimodalTimeSeriesDataset(
                DATASET_ROOT,
                ds_instance,
                DatasetSplit.VALIDATION,
                curr_modalities,
            ),
            test_transforms,
        )
        test_ds = AugmentedDataset(
            MultimodalTimeSeriesDataset(
                DATASET_ROOT,
                ds_instance,
                DatasetSplit.TEST,
                curr_modalities,
            ),
            test_transforms,
        )

        criterion = build_loss(cfg_loss)

        _, train_eval_loader, val_loader, test_loader = load_datasets(
            cfg_eval,
            core_train_ds,
            train_eval_ds,
            val_ds,
            test_ds,
            collate_fn,
            batch_sampler,
        )

        out_row = [run_name, run_id, epoch]

        train_eval_loss, acc, f1, precision, recall, ap = evaluate(
            model,
            train_eval_loader,
            criterion,
            device,
            batch_transforms_test,
        )
        out_row += eval_csv_rowpart(train_eval_loss, acc, f1, precision, recall, ap)

        val_loss, acc, f1, precision, recall, ap = evaluate(
            model,
            val_loader,
            criterion,
            device,
            batch_transforms_test,
        )
        out_row += eval_csv_rowpart(val_loss, acc, f1, precision, recall, ap)

        test_loss, acc, f1, precision, recall, ap = evaluate(
            model,
            test_loader,
            criterion,
            device,
            batch_transforms_test,
        )
        out_row += eval_csv_rowpart(test_loss, acc, f1, precision, recall, ap)

        with open(args.out_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(out_row)
