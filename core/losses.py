import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, reduction="mean", num_classes=None):
        super(FocalLoss, self).__init__()
        self._gamma = gamma
        self._weight = weight
        self._reduction = reduction
        self._num_classes = num_classes

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
