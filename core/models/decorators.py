import torch
import torch.nn as nn

from core.datasets.types import SparseSeriesDatasetSample


class PartiallyAnnotatedSemSegDecorator(nn.Module):
    def __init__(self, model: nn.Module):
        super(PartiallyAnnotatedSemSegDecorator, self).__init__()
        self.model = model

    def forward(self, x: SparseSeriesDatasetSample):
        logits = self.model(x)

        # flatten
        if logits.shape == x.target.shape:
            # meke oznake
            x.target = x.target.permute(0, 2, 3, 1).reshape(-1, x.target.shape[1])
            is_annotated = x.target.sum(1) == 1.0
        else:
            x.target = x.target.reshape(-1)
            is_annotated = x.target >= 0

        logits = logits.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])

        # filter
        logits = logits[is_annotated]
        x.target = x.target[is_annotated]

        return logits


if __name__ == "__main__":
    import lovely_tensors as lt
    from pprint import pprint

    lt.monkey_patch()

    model = PartiallyAnnotatedSemSegDecorator(lambda x: torch.randn((4, 2, 5, 5)))

    x = SparseSeriesDatasetSample(
        series=torch.randn((4, 3, 5, 5), dtype=torch.float),
        target=torch.randint(-1, 2, (4, 5, 5)),
        positions=torch.empty(1),
        ignore_mask=torch.empty(1),
    )
    pprint(x)

    logits = model(x)
    pprint(logits)
    pprint(x)
