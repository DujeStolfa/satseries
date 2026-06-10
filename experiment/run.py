import random
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


def run_experiment(
    cfg: ExperimentConfig,
    train_ds,
    val_ds,
    test_ds,
    collate_fn,
    device,
    batch_transforms,
):
    set_seed(cfg.training.seed)

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
        print("Train loss, acc, f1")
        print(train_loss, acc, f1)

        val_loss, acc, f1, conf_mat = evaluate(
            model,
            val_loader,
            criterion,
            device,
            batch_transforms,
        )
        print("Val loss, acc, f1")
        print(val_loss, acc, f1)

    test_loss, acc, f1, conf_mat = evaluate(
        model,
        test_loader,
        criterion,
        device,
        batch_transforms,
    )
    print("Test loss, acc, f1")
    print(test_loss, acc, f1)
