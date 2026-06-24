import torch.nn as nn

import core.models.presto_base as presto_base
from core.datasets import PrestoDatasetSample


class PrestoClassifier(nn.Module):
    def __init__(self, hidden_size, out_size, dropout, frozen):
        super().__init__()
        self.encoder = presto_base.Encoder(embedding_size=hidden_size, mlp_ratio=4)
        self.frozen = frozen
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=dropout)
        self.logits = nn.Linear(hidden_size, out_size)

    def forward(self, x: PrestoDatasetSample):
        embeddings = self.encoder(
            x=x.series,
            dynamic_world=x.dynamic_world,
            latlons=x.latlon,
            mask=x.mask,
            month=x.month,
            eval_task=True,
        )
        h = self.dropout(embeddings)
        logits = self.logits(h)
        return logits
