"""Utility helpers for logging, metrics, and directory management."""
import os


def make_directory(path: str) -> None:
    """Create directory if it does not exist."""
    try:
        os.makedirs(path)
    except OSError:
        pass


def print_args(args) -> None:
    """Print all arguments."""
    for arg in vars(args):
        print(f'{arg} {getattr(args, arg)}')


def mean(values: list) -> float:
    """Compute mean of a list of numbers."""
    return sum(values) / len(values)


def print_nparams(model) -> None:
    """Print the total number of parameters in a model."""
    nparams = sum(p.nelement() for p in model.parameters())
    print(f'Number of parameters: {nparams}')


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)