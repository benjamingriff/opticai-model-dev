import torch
from torch.utils.data import random_split


def split_dataset(dataset, val_split=0.2, seed=42):
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val], generator=generator)
