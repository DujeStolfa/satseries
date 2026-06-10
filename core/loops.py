import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm


def train(
    model: nn.Module,
    dataloader: data.DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device,
    clip,
    batch_transforms=None,
):
    model.train()
    train_loss = 0.0
    gt, preds = [], []

    import pdb

    for item, lengths in tqdm(dataloader, "Training", ncols=0):

        # pdb.set_trace()

        if batch_transforms is not None:
            item = batch_transforms(item)

        x = item.images[item.mask != -1].to(device)  # privremeno
        y = item.mask[item.mask != -1].to(device)

        model.zero_grad()

        logits = model(x)
        # pdb.set_trace()
        loss = criterion(logits, y.to(torch.uint8))
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        train_loss += loss.item()
        gt.append(y)
        preds.append(torch.argmax(logits, dim=1))

    gt = torch.cat(gt).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()

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

            x = item.images[item.mask != -1].to(device)
            y = item.mask[item.mask != -1].to(device)

            logits = model(x)
            eval_loss += criterion(logits, y.to(torch.uint8))

            gt.append(y)
            preds.append(torch.argmax(logits, dim=1))

    gt = torch.cat(gt).cpu().numpy()
    preds = torch.cat(preds).cpu().numpy()

    acc = accuracy_score(gt, preds)
    f1 = f1_score(gt, preds, average=None)
    confmat = confusion_matrix(gt, preds, normalize="true")
    return eval_loss / len(dataloader), acc, f1, confmat
