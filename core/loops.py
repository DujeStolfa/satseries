import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    average_precision_score,
)
from tqdm import tqdm


def _per_class_average_precision(gt, probs):
    return np.array(
        [average_precision_score(gt == i, probs[:, i]) for i in range(probs.shape[-1])]
    )


def train(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device,
    clip,
    batch_transforms=None,
):
    model.train()
    train_loss = 0.0
    gt, probs = [], []

    for item, lengths in tqdm(dataloader, "Training", ncols=0):
        if batch_transforms is not None:
            item = batch_transforms(item)

        item.to_device(device)
        model.zero_grad()

        logits = model(item)
        loss = criterion(logits, item.target)
        loss.backward()

        if clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        train_loss += loss.item()
        gt.append(item.target.detach().cpu())
        probs.append(torch.softmax(logits.detach(), dim=1).cpu())

    scheduler.step()

    probs = torch.cat(probs).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    gt = torch.cat(gt).cpu().numpy()
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)

    acc = accuracy_score(gt, preds)
    ap = _per_class_average_precision(gt, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(gt, preds, average=None)
    return train_loss / len(dataloader), acc, f1, precision, recall, ap


def evaluate(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    device,
    batch_transforms=None,
):
    model.eval()
    eval_loss = 0.0
    gt, probs = [], []

    with torch.no_grad():
        for item, lengths in tqdm(dataloader, "Evaluating", ncols=0):
            if batch_transforms is not None:
                item = batch_transforms(item)

            item.to_device(device)

            logits = model(item)
            eval_loss += criterion(logits, item.target).item()

            gt.append(item.target.detach().cpu())
            probs.append(torch.softmax(logits.detach().cpu(), dim=1))

    probs = torch.cat(probs).cpu().numpy()
    preds = np.argmax(probs, axis=1)

    gt = torch.cat(gt).cpu().numpy()
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)

    acc = accuracy_score(gt, preds)
    ap = _per_class_average_precision(gt, probs)
    precision, recall, f1, _ = precision_recall_fscore_support(gt, preds, average=None)
    return eval_loss / len(dataloader), acc, f1, precision, recall, ap
