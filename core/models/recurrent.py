import torch
import torch.nn as nn

from core.datasets import UnimodalDatasetSample
from core.models.modules import BahdanauAttentionPool, LinearGeluBN


class RecurrentModel(nn.Module):
    def __init__(
        self,
        rnn_cell: nn.Module,
        in_size,
        hidden_size,
        out_size,
        num_layers,
        head_cfg,
        dropout,
        bidirectional,
        attend: bool,
    ):
        super(RecurrentModel, self).__init__()

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
            self.attn_pool = BahdanauAttentionPool(
                in_size=d * hidden_size,
                hidden_size=(d * hidden_size) // 2,
                batch_first=False,
            )
        else:
            self.attn_pool = None

        head_in_size = d * hidden_size
        if attend:
            # Konkateniraj izlaz paznje s izlazom iz povratnog modula
            head_in_size *= 2

        head_cfg = [head_in_size] + head_cfg

        self.blocks = nn.ModuleList(
            [
                LinearGeluBN(block_in, block_out, dropout)
                for block_in, block_out in zip(head_cfg[:-1], head_cfg[1:])
            ]
        )
        self.logits = nn.Linear(in_features=head_cfg[-1], out_features=out_size)

    def forward(self, x: UnimodalDatasetSample):
        # Transponiraj ulaze [B, T, H] > [T, B, H]
        x = torch.transpose(x.series, 0, 1)

        # TODO: pack_padded_sequence?

        # Skrivena stanja posljednjeg sloja
        #   - h_last_layer - [T, B, d * hidden_size]
        # Skrivena stanja u posljednjem koraku
        #   - h_final_tstep - [d * num_layers, B, hidden_size]
        h_last_layer, h_final_tstep = self.rnn(x)

        if self.attn_pool:
            # out_attn - [B, d * hidden_size]
            out_attn, alpha = self.attn_pool(None, h_last_layer, h_last_layer)

            # h_dec_t = [h_dec_t ; out_attn]
            h = torch.cat([h_last_layer[-1], out_attn], dim=1)
        else:
            # Za dvosmjerni model:
            # - zadnji sloj (iz svakog smjera) => [B, 2 * hidden_size]
            # - isto kao i torch.cat((h_final_tstep[-1], h_tfinal_step[-2]), dim=1)
            h = h_last_layer[-1]

        for block in self.blocks:
            h = block(h)

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
    model = RecurrentModel(
        nn.LSTM,
        in_size,
        hidden_size,
        out_size,
        num_layers,
        [],
        0,
        bidirectional=True,
        attend=False,
    )

    logits = model(batch)
    print(logits)
