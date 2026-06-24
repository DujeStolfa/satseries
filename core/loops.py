import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


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
    gt, preds = [], []

    for item, lengths in tqdm(dataloader, "Training", ncols=0):
        if batch_transforms is not None:
            item = batch_transforms(item)

        item.to_device(device)
        model.zero_grad()

        logits = model(item)
        loss = criterion(logits, item.target)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        gt.append(item.target)
        preds.append(torch.argmax(logits, dim=1))

    preds = torch.cat(preds).cpu().numpy()
    gt = torch.cat(gt).cpu().numpy()
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)

    acc = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average=None)
    return train_loss / len(dataloader), acc, f1


def evaluate(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    device,
    batch_transforms=None,
):
    model.eval()
    eval_loss = 0.0
    gt, preds = [], []

    with torch.no_grad():
        for item, lengths in tqdm(dataloader, "Evaluating", ncols=0):
            if batch_transforms is not None:
                item = batch_transforms(item)

            item.to_device(device)

            logits = model(item)
            eval_loss += criterion(logits, item.target)

            gt.append(item.target)
            preds.append(torch.argmax(logits, dim=1))

    preds = torch.cat(preds).cpu().numpy()
    gt = torch.cat(gt).cpu().numpy()
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)

    acc = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average=None)
    return eval_loss / len(dataloader), acc, f1
