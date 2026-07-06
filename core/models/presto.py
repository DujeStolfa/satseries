import torch.nn as nn

import core.models.presto_base as presto_base
from core.datasets import PrestoDatasetSample


class PrestoLinear(nn.Module):
    def __init__(self, hidden_size, out_size, dropout, frozen):
        super().__init__()
        self.encoder = presto_base.Encoder(embedding_size=hidden_size, mlp_ratio=4)
        self.frozen = frozen

        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)

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


class _LinearGeluBN(nn.Sequential):
    def __init__(self, in_size, out_size, dropout):
        super(_LinearGeluBN, self).__init__()
        self.append(nn.Linear(in_size, out_size))
        self.append(nn.BatchNorm1d(out_size))
        self.append(nn.GELU())
        self.append(nn.Dropout(p=dropout))


class PrestoDeep(nn.Module):
    def __init__(self, hidden_size, out_size, head_cfg, dropout, frozen):
        super().__init__()
        self.encoder = presto_base.Encoder(embedding_size=hidden_size, mlp_ratio=4)
        self.frozen = frozen

        self.encoder.pos_embed.requires_grad_(False)
        self.encoder.month_embed.requires_grad_(False)

        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False

        head_cfg = [hidden_size] + head_cfg
        self.blocks = nn.ModuleList(
            [
                _LinearGeluBN(block_in, block_out, dropout)
                for block_in, block_out in zip(head_cfg[:-1], head_cfg[1:])
            ]
        )
        self.logits = nn.Linear(head_cfg[-1], out_size)

    def forward(self, x: PrestoDatasetSample):
        embeddings = self.encoder(
            x=x.series,
            dynamic_world=x.dynamic_world,
            latlons=x.latlon,
            mask=x.mask,
            month=x.month,
            eval_task=True,
        )
        h = embeddings
        for block in self.blocks:
            h = block(h)
        logits = self.logits(h)
        return logits
