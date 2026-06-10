import random
from dataclasses import asdict

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from core.models import build_model
from core.loops import train, evaluate
from experiment.configs import ExperimentConfig


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Doesn't set seeds for core.transforms


def load_datasets(cfg: ExperimentConfig, train_ds, val_ds, test_ds, collate_fn):
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size_test,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.training.batch_size_test,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def log_eval(split: str, loss, acc, f1, **kwargs):
    mlflow.log_metric(f"{split}_loss", loss, **kwargs)
    mlflow.log_metric(f"{split}_acc", acc, **kwargs)
    mlflow.log_metric(f"{split}_f1_macro", f1.mean(), **kwargs)
    mlflow.log_metric(f"{split}_f1_worst", f1.min(), **kwargs)

    for i, v in enumerate(f1):
        mlflow.log_metric(f"{split}_f1_class_{i}", v, **kwargs)


def run_experiment(
    run_name,
    cfg: ExperimentConfig,
    train_ds,
    val_ds,
    test_ds,
    collate_fn,
    device,
    batch_transforms,
):
    set_seed(cfg.training.seed)
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment(cfg.name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": cfg.model["name"],
                **{f"model.{k}": v for k, v in cfg.model.items() if k != "name"},
            }
        )
        for key, cfg_t in cfg.transforms.items():
            mlflow.log_params(
                {
                    **{
                        f"transforms.{key}.{cfg_t["name"]}.{k}": v
                        for k, v in cfg_t.items()
                        if k != "name"
                    },
                }
            )
        mlflow.log_params(
            {
                **{f"data.{k}": v for k, v in cfg.data.items() if k != "name"},
            }
        )
        mlflow.log_params(asdict(cfg.training))

        train_loader, val_loader, test_loader = load_datasets(
            cfg, train_ds, val_ds, test_ds, collate_fn
        )

        model = build_model(cfg.model).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.wd,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(cfg.training.epochs):
            print(f"\nEpoch {epoch + 1}")

            train_loss, acc, f1 = train(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                cfg.training.clip,
                batch_transforms,
            )
            log_eval("train", train_loss, acc, f1, step=epoch)

            val_loss, acc, f1, conf_mat = evaluate(
                model,
                val_loader,
                criterion,
                device,
                batch_transforms,
            )
            log_eval("val", val_loss, acc, f1, step=epoch)

        test_loss, acc, f1, conf_mat = evaluate(
            model,
            test_loader,
            criterion,
            device,
            batch_transforms,
        )
        log_eval("test", test_loss, acc, f1, step=epoch)

        mlflow.pytorch.log_model(model, name="model")
        return model
