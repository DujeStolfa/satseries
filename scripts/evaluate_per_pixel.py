import csv

from sklearn.metrics import f1_score
import torch
from mlflow.pytorch import load_model
from tqdm import tqdm

from core.datasets import (
    multimodal_pad_collate_fn,
    DatasetInstance,
    DatasetSplit,
    Modality,
    MultimodalTimeSeriesDataset,
    AugmentedDataset,
)
import core.transforms as t


def apply_mask(x, mask):
    for ts in x.modalities.values():
        ts.images = ts.images[mask]
        ts.timesteps = ts.timesteps[mask]

    x.latlon = x.latlon[mask]
    x.target = x.target[mask]
    return x


if __name__ == "__main__":
    out_file = "/mnt/teratron/rezultati/per_pixel_eval.csv"

    model = load_model(
        "runs:/30c9755e1a56455fac76377e24f0cca4/model"
    )
    print(model.to("cuda"))

    SEED = 1414213
    DATASET_ROOT = "/mnt/teratron/data/amorfa"
    collate_fn = multimodal_pad_collate_fn
    ds_instance = DatasetInstance.REGIONAL
    curr_modalities = [Modality.SENTINEL_2_L2A, Modality.SENTINEL_1_ASC]

    label_mapper = torch.Tensor(
        [
            [1, 0],
            [0, 1],
            [0, 1],
            [0.25, 0.75],
            [0.75, 0.25],
        ]
    )

    transforms = t.Compose(
        [
            t.ApplyToModality(
                t.MonthlyRandomSample(
                    month_start=4,
                    month_end=10,
                    seed=SEED,  # reduction="median",  #
                ),
                curr_modalities,
            ),
            t.ToTensor(),
            t.ApplyToModality(
                t.Compose(
                    [
                        t.Scale(1e-4, dim=1),
                        t.AddSpectralFeatures(dim=1, r_idx=3, nir_idx=7),
                    ]
                ),
                Modality.SENTINEL_2_L2A,
            ),
            t.ApplyToModality(
                t.Compose([t.Translate(25, dim=1), t.Scale(1 / 25, dim=1)]),
                Modality.SENTINEL_1_ASC,
            ),
            # t.Normalize(mean=data_stats["mean"], std=data_stats["std"]),
        ]
    )
    spatial_flatten = t.BatchSpatialFlatten(batch_first=True)
    batch_transforms = t.Compose(
        [
            t.ToPrestoFormat(month_start=4),
            t.MapLabels(mapper=label_mapper),
        ]
    )

    criterion = torch.nn.CrossEntropyLoss()

    with open(out_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "item_idx",
                "series_id",
                "f1_macro",
                "loss",
                "pixel_idx",
                "gt",
                "pred",
                "prob",
                "correct",
            ]
        )

    with torch.no_grad():
        model.eval()

        for split in [DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST]:
            print("")
            print("Split:", split.value)

            ds = AugmentedDataset(
                MultimodalTimeSeriesDataset(
                    DATASET_ROOT,
                    ds_instance,
                    split,
                    curr_modalities,
                ),
                transforms,
            )

            for i, item in tqdm(enumerate(ds), "Evaluating", ncols=0):

                # batch transforms
                batch, _ = multimodal_pad_collate_fn([item])

                batch = spatial_flatten(batch)

                valid_mask = batch.target != -1
                indices = torch.argwhere(valid_mask).squeeze()

                batch = apply_mask(batch, valid_mask)
                batch = batch_transforms(batch)

                # forward
                batch.to_device("cuda")
                logits = model(batch)
                loss = criterion(logits, batch.target).item()

                # metrics
                gt = torch.argmax(batch.target, dim=1).detach().cpu()
                preds = torch.argmax(logits.detach().cpu(), dim=1)
                f1 = f1_score(gt, preds, average="macro")

                probs = torch.softmax(logits, dim=1).detach().cpu()
                b = probs.shape[0]
                probs = probs[torch.arange(b), gt]
                gt_probs = batch.target[torch.arange(b), 1].detach().cpu()

                # image level
                ones = torch.ones((b), dtype=torch.float)
                f1 = ones * f1
                idx = ones * i
                sid = ones * ds._dataset._series_ids[i]
                loss = ones * loss

                with open(out_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerows(
                        zip(
                            [split.value] * b,
                            idx.tolist(),
                            sid.tolist(),
                            f1.tolist(),
                            loss.tolist(),
                            indices.tolist(),
                            gt_probs.tolist(),
                            preds.tolist(),
                            probs.tolist(),
                            (gt == preds).tolist(),
                        )
                    )
