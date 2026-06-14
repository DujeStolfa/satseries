import torch.optim as optim
import torch.optim.lr_scheduler as sched


def build_optimizer(params, cfg: dict) -> optim.Optimizer:
    name = cfg.pop("name")
    cls = getattr(optim, name)
    return cls(params, **cfg)


def build_scheduler(optimizer, cfg: dict) -> sched.LRScheduler:
    name = cfg.pop("name")
    cls = getattr(sched, name)
    return cls(optimizer, **cfg)
