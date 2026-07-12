"""
Prilagodeni moduli `Temporal Attention Encoder` i `Lightweight Temporal Attention Encoder` iz
- https://github.com/VSainteuf/pytorch-psetae/tree/master
- https://github.com/VSainteuf/lightweight-temporal-attention-pytorch/tree/master

Credits:
The module is heavily inspired by the works of Vaswani et al. on self-attention and their pytorch implementation of
the Transformer served as code base for the present script.

paper: https://arxiv.org/abs/1706.03762
code: github.com/jadore801120/attention-is-all-you-need-pytorch
"""

import torch
import torch.nn as nn
import numpy as np
import copy

from core.datasets.types import SparseSeriesDatasetSample


class TemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=4,
        d_k=32,
        d_model=None,
        n_neurons=[512, 128, 128],
        dropout=0.2,
        T=1000,
        len_max_seq=365,
    ):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model

        """

        super(TemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = copy.deepcopy(n_neurons)

        self.name = "TAE_dk{}_{}Heads_{}_T{}_do{}".format(
            d_k, n_head, "|".join(list(map(str, self.n_neurons))), T, dropout
        )

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(len_max_seq + 1, self.in_channels, T=T),
            freeze=True,
        )

        self.inlayernorm = nn.LayerNorm(self.in_channels)

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
            self.inconv_ln = nn.LayerNorm(d_model)
            self.name += "_dmodel{}".format(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        self.outlayernorm = nn.LayerNorm(self.d_model)

        self.attention_heads = GlobalQueryAttentionPool(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        assert self.n_neurons[0] == n_head * self.d_model
        assert self.n_neurons[-1] == self.d_model
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                    nn.BatchNorm1d(self.n_neurons[i + 1]),
                    nn.ReLU(),
                ]
            )

        self.mlp = nn.Sequential(*layers)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: SparseSeriesDatasetSample):
        batch_size, seq_len, d = x.series.shape

        h = self.inlayernorm(x.series)
        enc_output = h + self.position_enc(x.positions)

        if self.inconv is not None:
            enc_output = self.inconv(enc_output.permute(0, 2, 1)).permute(0, 2, 1)
            enc_output = self.inconv_ln(enc_output)

        enc_output, attn = self.attention_heads(
            enc_output, enc_output, enc_output, mask=~x.ignore_mask
        )

        enc_output = (
            enc_output.permute(1, 0, 2).contiguous().view(batch_size, -1)
        )  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        return enc_output


class LightweightTemporalAttentionEncoder(nn.Module):
    def __init__(
        self,
        in_channels=128,
        n_head=16,
        d_k=8,
        n_neurons=[256, 128],
        dropout=0.2,
        d_model=256,
        T=1000,
        len_max_seq=365,
        return_att=False,
    ):
        """
        Sequence-to-embedding encoder.
        Args:
            in_channels (int): Number of channels of the input embeddings
            n_head (int): Number of attention heads
            d_k (int): Dimension of the key and query vectors
            n_neurons (list): Defines the dimensions of the successive feature spaces of the MLP that processes
                the concatenated outputs of the attention heads
            dropout (float): dropout
            T (int): Period to use for the positional encoding
            len_max_seq (int, optional): Maximum sequence length, used to pre-compute the positional encoding table
            d_model (int, optional): If specified, the input tensors will first processed by a fully connected layer
                to project them into a feature space of dimension d_model
            return_att (bool): If true, the module returns the attention masks along with the embeddings (default False)

        """
        super(LightweightTemporalAttentionEncoder, self).__init__()
        self.in_channels = in_channels
        self.n_neurons = copy.deepcopy(n_neurons)
        self.return_att = return_att

        if d_model is not None:
            self.d_model = d_model
            self.inconv = nn.Conv1d(in_channels, d_model, 1)
            self.inconv_ln = nn.LayerNorm(d_model)
        else:
            self.d_model = in_channels
            self.inconv = None

        sin_tab = get_sinusoid_encoding_table(
            len_max_seq + 1, self.d_model // n_head, T=T
        )
        self.position_enc = nn.Embedding.from_pretrained(
            torch.cat([sin_tab for _ in range(n_head)], dim=1), freeze=True
        )

        self.inlayernorm = nn.LayerNorm(self.in_channels)
        self.outlayernorm = nn.LayerNorm(n_neurons[-1])
        self.attention_heads = LightweightAttentionPool(
            n_head=n_head, d_k=d_k, d_in=self.d_model
        )

        assert self.n_neurons[0] == self.d_model
        activation = nn.ReLU()
        layers = []
        for i in range(len(self.n_neurons) - 1):
            layers.extend(
                [
                    nn.Linear(self.n_neurons[i], self.n_neurons[i + 1]),
                    nn.BatchNorm1d(self.n_neurons[i + 1]),
                    activation,
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: SparseSeriesDatasetSample):
        batch_size, seq_len, d = x.series.shape
        h = self.inlayernorm(x.series)

        if self.inconv is not None:
            h = self.inconv(h.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.inconv_ln(h)

        enc_output = h + self.position_enc(x.positions)
        enc_output, attn = self.attention_heads(enc_output, mask=~x.ignore_mask)

        enc_output = (
            enc_output.permute(1, 0, 2).contiguous().view(batch_size, -1)
        )  # Concatenate heads

        enc_output = self.outlayernorm(self.dropout(self.mlp(enc_output)))

        if self.return_att:
            return enc_output, attn
        else:
            return enc_output


class GlobalQueryAttentionPool(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.fc1_q = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_q.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(n_head * d_k), nn.Linear(n_head * d_k, n_head * d_k)
        )

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, q, k, v, mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        batch_size, seq_len, _ = q.size()

        q = self.fc1_q(q).view(batch_size, seq_len, n_head, d_k)

        # MEAN query with mask
        weights = mask.float()
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1)
        q = (q * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)

        q = self.fc2(q.view(batch_size, n_head * d_k)).view(batch_size, n_head, d_k)
        q = q.permute(1, 0, 2).contiguous().view(n_head * batch_size, d_k)

        k = self.fc1_k(k).view(batch_size, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = v.repeat(n_head, 1, 1)  # (n*b) x lv x d_in

        mask = mask.repeat_interleave(n_head, dim=0)
        output, attn = self.attention(q, k, v, mask=mask)
        output = output.view(n_head, batch_size, 1, d_in)
        output = output.squeeze(dim=2)

        return output, attn


class LightweightAttentionPool(nn.Module):
    def __init__(self, n_head, d_k, d_in):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in

        self.Q = nn.Parameter(torch.zeros((n_head, d_k))).requires_grad_(True)
        nn.init.normal_(self.Q, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.fc1_k = nn.Linear(d_in, n_head * d_k)
        nn.init.normal_(self.fc1_k.weight, mean=0, std=np.sqrt(2.0 / (d_k)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

    def forward(self, x, mask=None):
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        batch_size, seq_len, _ = x.size()

        q = self.Q[:, None, :].expand(-1, batch_size, -1)
        q = q.reshape(-1, d_k)  # (n*b) x d_k

        k = self.fc1_k(x).view(batch_size, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n*b) x lk x dk

        v = torch.stack(x.split(x.shape[-1] // n_head, dim=-1)).view(
            n_head * batch_size, seq_len, -1
        )

        mask = mask.repeat_interleave(n_head, dim=0)
        output, attn = self.attention(q, k, v, mask=mask)
        attn = attn.view(n_head, batch_size, 1, seq_len)
        attn = attn.squeeze(dim=2)

        output = output.view(n_head, batch_size, 1, d_in // n_head)
        output = output.squeeze(dim=2)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q.unsqueeze(1), k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            # Only attend to tokens with mask == 1
            attn = attn.masked_fill(mask.unsqueeze(1) == 0, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)

        return output, attn


def get_sinusoid_encoding_table(positions, d_hid, T=1000):
    """Sinusoid position encoding table
    positions: int or list of integer, if int range(positions)"""

    if isinstance(positions, int):
        positions = list(range(positions))

    def cal_angle(position, hid_idx):
        return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if torch.cuda.is_available():
        return torch.FloatTensor(sinusoid_table).cuda()
    else:
        return torch.FloatTensor(sinusoid_table)


if __name__ == "__main__":
    import lovely_tensors as lt

    lt.monkey_patch()

    B, T, C = 3, 4, 5

    mask = torch.zeros((B, T), dtype=torch.bool)
    mask[0, -1] = 1
    mask[2, -2:] = 1

    sample = SparseSeriesDatasetSample(
        series=torch.randn((B, T, C)),
        target=torch.ones(B),
        positions=torch.tensor([5, 7, 26, 30]),
        ignore_mask=mask,
    )
    sample.to_device("cuda")

    tae = TemporalAttentionEncoder(
        in_channels=C,
        n_head=2,
        d_k=32,
        d_model=16,
        n_neurons=[2 * 16, 16],
    )
    tae.to("cuda")
    out_tae = tae(sample)

    print("TAE", out_tae)

    ltae = LightweightTemporalAttentionEncoder(
        in_channels=C,
        n_head=2,
        d_k=8,
        d_model=16,
        n_neurons=[16, 16],
    )
    ltae.to("cuda")

    out_ltae = ltae(sample)
    print("LTAE", out_ltae)

    import pdb

    pdb.set_trace()
