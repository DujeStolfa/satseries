import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._weight = weight
        self._reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        p_t = torch.exp(-ce_loss)
        focal_weight = (1 - p_t) ** self._gamma

        loss = focal_weight * ce_loss

        if self._weight is not None:
            alpha_t = self._weight.to(inputs.device)[targets]
            loss = alpha_t * loss

        if self._reduction == "mean":
            if self._weight is not None:
                return loss.sum() / alpha_t.sum()
            return loss.mean()

        elif self._reduction == "sum":
            return loss.sum()

        return loss


class SymmetricCrossEntropyLoss(torch.nn.Module):
    """https://github.com/HanxunH/SCELoss-Reproduce/tree/master"""

    def __init__(self, alpha, beta, num_classes, weight=None, reduction="mean"):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self._alpha = alpha
        self._beta = beta
        self._num_classes = num_classes
        self._weight = weight
        self._reduction = reduction

    def forward(self, pred, labels):
        # CCE
        ce = F.cross_entropy(
            pred, labels, weight=self._weight, reduction=self._reduction
        )

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = (
            torch.nn.functional.one_hot(labels, self._num_classes)
            .float()
            .to(pred.device)
        )
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = -1 * torch.sum(pred * torch.log(label_one_hot), dim=1)

        if self._reduction == "mean":
            rce = rce.mean()
        elif self._reduction == "sum":
            rce = rce.sum()

        # Loss
        loss = self._alpha * ce + self._beta * rce
        return loss


def build_loss(cfg: dict) -> nn.Module:
    name = cfg.pop("name")

    if name == "FocalLoss":
        return FocalLoss(**cfg)

    if name == "SymmetricCrossEntropyLoss":
        return SymmetricCrossEntropyLoss(**cfg)

    cls = getattr(nn, name)
    return cls(**cfg)
