import torch
import torch.nn as nn

from core.datasets.types import SparseSeriesDatasetSample
from core.models.modules import LinearGeluBN
from core.models.presto_base import get_sinusoid_encoding_table


class SequenceClassificationTransformer(nn.Module):
    def __init__(
        self,
        in_size,
        embed_size,
        out_size,
        num_layers,
        num_heads,
        head_cfg,
        dropout,
        attention_dropout,
    ):
        super(SequenceClassificationTransformer, self).__init__()

        self.layer_norm_in = nn.LayerNorm(in_size)
        self.embed = nn.Linear(in_size, embed_size)

        self.encoder = TransformerEncoder(
            embed_size,
            num_layers,
            num_heads,
            dropout,
            attention_dropout,
            max_seq_length=365,
            T=1000,
        )

        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        nn.init.trunc_normal_(self.class_token, std=0.02)

        head_cfg = [embed_size] + head_cfg
        self.blocks = nn.ModuleList(
            [
                LinearGeluBN(block_in, block_out, dropout)
                for block_in, block_out in zip(head_cfg[:-1], head_cfg[1:])
            ]
        )

        self.logits = nn.Linear(in_features=head_cfg[-1], out_features=out_size)

    def forward(self, x: SparseSeriesDatasetSample):
        batch_size, t, _ = x.series.shape

        h = self.embed(self.layer_norm_in(x.series))

        # CLS token
        h = torch.cat([self.class_token.expand(batch_size, -1, -1), h], dim=1)

        cls_mask = torch.zeros(
            (batch_size, 1), device=x.ignore_mask.device, dtype=torch.bool
        )
        mask = torch.cat([cls_mask, x.ignore_mask.to(torch.bool)], dim=1)

        cls_position = torch.zeros(
            (batch_size, 1), device=x.positions.device, dtype=torch.long
        )
        positions = torch.cat([cls_position, x.positions + 1], dim=1)

        h = self.encoder(h, positions, mask)
        h = h[:, 0]

        for block in self.blocks:
            h = block(h)

        logits = self.logits(h)
        return logits


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        in_size,
        num_layers,
        num_heads,
        dropout,
        attention_dropout,
        max_seq_length=365,
        T=1000,
    ):
        super(TransformerEncoder, self).__init__()

        self.pos_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_seq_length + 1, in_size, T=T),
            freeze=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(in_size, num_heads, dropout, attention_dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln = nn.LayerNorm(in_size)

    def forward(self, series, positions, ignore_mask):
        h = series + self.pos_embedding(positions)
        h = self.dropout(h)

        for encoder_layer in self.layers:
            h = encoder_layer(h, ignore_mask)

        h = self.ln(h)
        return h


class TransformerEncoderLayer(nn.Module):
    """https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py"""

    def __init__(self, in_size, num_heads, dropout, attention_dropout):
        super(TransformerEncoderLayer, self).__init__()

        # Samopaznja
        self.norm1 = nn.LayerNorm(in_size)
        self.self_attention = nn.MultiheadAttention(
            in_size, num_heads, attention_dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # FFN
        self.norm2 = nn.LayerNorm(in_size)
        self.ffn = nn.Sequential(
            nn.Linear(in_size, 4 * in_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * in_size, in_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, input, ignore_mask):
        x = self.norm1(input)
        x, _ = self.self_attention(x, x, x, key_padding_mask=ignore_mask)
        x = self.dropout1(x)
        x = input + x

        y = self.norm2(x)
        y = self.ffn(y)
        y = self.dropout2(y)
        y = x + y

        return y
