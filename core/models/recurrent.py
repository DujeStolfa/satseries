import torch
import torch.nn as nn

from core.models.modules import BahdanauAttention


class RNNModel(nn.Module):
    def __init__(
        self,
        rnn_cell: nn.Module,
        in_size,
        hidden_size,
        out_size,
        num_layers,
        dropout,
        bidirectional,
        attend: bool,
    ):
        super().__init__()
        # rnn(hidden_size) x num_layers -> fc(hidden_size, hidden_size) ->
        # -> ReLU() -> fc(hidden_size, out_size)

        self.rnn = rnn_cell(
            input_size=in_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        d = 2 if bidirectional else 1

        if attend:
            self.attn_pool = BahdanauAttention(
                in_size=d * hidden_size,
                out_size=(d * hidden_size) // 2,
                batch_first=False,
            )
        else:
            self.attn_pool = None

        self.fc3 = nn.Linear(in_features=d * hidden_size, out_features=hidden_size)
        self.relu3 = nn.ReLU()
        self.logits = nn.Linear(in_features=hidden_size, out_features=out_size)

    def forward(self, x):
        # Transponiraj ulaze [B, T, H] > [T, B, H]
        x = torch.transpose(x.images, 0, 1)

        # TODO: pack_padded_sequence?

        out, h = self.rnn(x)
        if self.attn_pool:
            h, alpha = self.attn_pool(None, out, out)  # [B, C]
        else:
            if isinstance(self.rnn, nn.LSTM):
                h, c = h

            # Zadnji sloj (iz svakog smjera) => [B, 2 * C]
            if self.rnn.bidirectional:
                h = torch.cat((h[-2], h[-1]), dim=1)
            else:
                h = h[-1]

        h = self.fc3(h)
        h = self.relu3(h)

        logits = self.logits(h)
        return torch.squeeze(logits)


if __name__ == "__main__":
    import lovely_tensors as lt

    lt.monkey_patch()

    in_size = 5
    hidden_size = 9
    out_size = 3
    num_layers = 2
    t = 12
    b = 7

    batch = torch.rand((t, b, in_size))
    print(batch)
    model = RNNModel(
        nn.LSTM,
        in_size,
        hidden_size,
        out_size,
        num_layers,
        0,
        bidirectional=True,
        attend=False,
    )

    logits = model(batch)
    print(logits)
