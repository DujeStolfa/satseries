import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from tqdm import tqdm

from core.evaluation import Evaluator


def train(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    evaluator: Evaluator,
    device,
    clip,
    batch_transforms=None,
):
    model.train()
    evaluator.reset()
    train_loss = 0.0

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
        evaluator.update(
            item.target.detach().cpu(),
            torch.softmax(logits.detach().cpu(), dim=1).numpy(),
        )

    scheduler.step()

    return train_loss / len(dataloader), evaluator.evaluate()


def evaluate(
    model: nn.Module,
    dataloader: data.DataLoader,
    criterion: nn.Module,
    evaluator: Evaluator,
    device,
    batch_transforms=None,
):
    model.eval()
    evaluator.reset()
    eval_loss = 0.0

    with torch.no_grad():
        for item, lengths in tqdm(dataloader, "Evaluating", ncols=0):
            if batch_transforms is not None:
                item = batch_transforms(item)

            item.to_device(device)

            logits = model(item)
            eval_loss += criterion(logits, item.target).item()

            evaluator.update(
                item.target.detach().cpu(),
                torch.softmax(logits.detach().cpu(), dim=1).numpy(),
            )

    return eval_loss / len(dataloader), evaluator.evaluate()
