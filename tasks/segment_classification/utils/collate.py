import torch
from torch.nn.utils.rnn import pad_sequence
from labels.phases import phase2idx


def uniform_sample(tensor, num_frames):
    T = tensor.shape[0]
    if T <= num_frames:
        return tensor
    idxs = torch.linspace(0, T - 1, steps=num_frames).long()
    return tensor[idxs]


def segment_collate(batch, num_frames=200):
    """
    Collate function for segment-based datasets.
    Converts normalized labels (strings) to indices using phase2idx.
    """
    segments, labels = zip(*batch)
    sampled_segments = [uniform_sample(s, num_frames) for s in segments]
    lengths = [s.shape[0] for s in sampled_segments]
    padded_segments = pad_sequence(sampled_segments, batch_first=True)
    label_indices = [phase2idx[label] for label in labels]
    return padded_segments, torch.tensor(label_indices), lengths
