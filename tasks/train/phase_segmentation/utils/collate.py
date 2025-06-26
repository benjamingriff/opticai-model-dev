import torch
from labels.phases import phase2idx


def collate_mstcn(batch):
    sequences, labels = zip(*batch)
    x = torch.stack(sequences)
    idx_labels = []
    for label_seq in labels:
        idx_seq = [phase2idx[label] for label in label_seq]
        idx_labels.append(torch.tensor(idx_seq, dtype=torch.long))
    y = torch.stack(idx_labels)
    return x, y


def select_collate(model_type):
    if model_type == "ms_tcn":
        return collate_mstcn
    else:
        raise ValueError(f"Unknown model type: {model_type}")
