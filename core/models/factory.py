import torch.nn as nn

from core.models.presto import PrestoClassifier
from core.models.recurrent import RNNModel


def build_model(cfg: dict) -> nn.Module:
    name = cfg["name"]

    if name == "recurrent":
        cell_map = {
            "rnn": nn.RNN,
            "gru": nn.GRU,
            "lstm": nn.LSTM,
        }
        return RNNModel(
            cell_map[cfg["rnn_cell"]],
            cfg["in_size"],
            cfg["hidden_size"],
            cfg["out_size"],
            cfg["num_layers"],
            cfg["dropout"],
            cfg["bidirectional"],
            cfg["attend"],
        )

    if name == "presto":
        return PrestoClassifier(
            cfg["hidden_size"],
            cfg["out_size"],
            cfg["dropout"],
        )

    raise ValueError(f"Unknown model: {name}")
