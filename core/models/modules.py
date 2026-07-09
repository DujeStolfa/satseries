import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttentionPool(nn.Module):
    def __init__(self, in_size, hidden_size, batch_first):
        super(BahdanauAttentionPool, self).__init__()
        self.W1 = nn.Linear(in_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, 1)
        self.batch_first = batch_first

    def forward(self, queries, keys, values):
        # [T, B, C] -> [1, B, C]
        if queries is not None:
            x = torch.cat((queries, keys), -1)
        else:
            x = keys

        if not self.batch_first:
            x = x.transpose(0, 1)
            values = values.transpose(0, 1)

        a = self.w2(torch.tanh(self.W1(x)))
        a = a.transpose(1, 2)
        alpha = F.softmax(a, dim=-1)
        out_attn = torch.bmm(alpha, values).squeeze(1)

        return out_attn, alpha


class LinearGeluBN(nn.Sequential):
    def __init__(self, in_size, out_size, dropout):
        super(LinearGeluBN, self).__init__()
        self.append(nn.Linear(in_size, out_size))
        self.append(nn.BatchNorm1d(out_size))
        self.append(nn.GELU())
        self.append(nn.Dropout(p=dropout))
