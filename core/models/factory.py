import torch.nn as nn

from core.models.presto import PrestoLinear, PrestoDeep
from core.models.recurrent import RecurrentModel
from core.models.temporal_attention import LtaeClassifier, TaeClassifier
from core.models.transformer import SequenceClassificationTransformer


def build_model(cfg: dict) -> nn.Module:
    name = cfg["name"]

    if name == "recurrent":
        cell_map = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        return RecurrentModel(
            cell_map[cfg["rnn_cell"]],
            cfg["in_size"],
            cfg["hidden_size"],
            cfg["out_size"],
            cfg["num_layers"],
            cfg["head"],
            cfg["dropout"],
            cfg["bidirectional"],
            cfg["attend"],
        )

    if name == "presto":
        return PrestoLinear(
            cfg["hidden_size"],
            cfg["out_size"],
            cfg["dropout"],
            cfg["frozen"],
        )

    if name == "presto_deep":
        return PrestoDeep(
            cfg["hidden_size"],
            cfg["out_size"],
            cfg["head"],
            cfg["dropout"],
            cfg["frozen"],
        )

    if name == "tae":
        return TaeClassifier(
            cfg["in_size"],
            cfg["hidden_size"],
            cfg["embedd_dim"],
            cfg["num_heads"],
            cfg["out_size"],
            cfg["head"],
            cfg["dropout"],
        )

    if name == "ltae":
        return LtaeClassifier(
            cfg["in_size"],
            cfg["hidden_size"],
            cfg["embedd_dim"],
            cfg["num_heads"],
            cfg["out_size"],
            cfg["head"],
            cfg["dropout"],
        )

    if name == "transformer":
        return SequenceClassificationTransformer(
            cfg["in_size"],
            cfg["embed_size"],
            cfg["out_size"],
            cfg["num_layers"],
            cfg["num_heads"],
            cfg["head"],
            cfg["dropout"],
            cfg["attn_dropout"],
        )

    raise ValueError(f"Unknown model: {name}")
