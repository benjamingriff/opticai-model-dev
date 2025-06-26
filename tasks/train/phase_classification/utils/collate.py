import torch
from torch.nn.utils.rnn import pad_sequence
from labels.phases import phase2idx
from functools import partial


def uniform_sample(tensor, num_frames):
    T = tensor.shape[0]
    if T >= num_frames:
        idxs = torch.linspace(0, T - 1, steps=num_frames).long()
        return tensor[idxs]
    else:
        pad_len = num_frames - T
        pad = tensor[-1].unsqueeze(0).repeat(pad_len, 1, 1, 1)
        return torch.cat([tensor, pad], dim=0)


# def uniform_sample(tensor, num_frames):
#     """Uniformly sample or truncate a segment to num_frames."""
#     T = tensor.shape[0]
#     if T <= num_frames:
#         return tensor
#     idxs = torch.linspace(0, T - 1, steps=num_frames).long()
#     return tensor[idxs]


def collate_cnn_lstm(batch, num_frames):
    """Collate function for CNN + LSTM models."""
    segments, labels = zip(*batch)
    sampled_segments = [uniform_sample(s, num_frames) for s in segments]
    lengths = [s.shape[0] for s in sampled_segments]
    padded_segments = pad_sequence(sampled_segments, batch_first=True)
    label_indices = [phase2idx[label] for label in labels]
    return padded_segments, torch.tensor(label_indices), lengths


def collate_cnn_tcn(batch, num_frames):
    """Collate function for CNN + TCN models."""
    segments, labels = zip(*batch)
    sampled_segments = [uniform_sample(s, num_frames) for s in segments]
    segment_tensor = torch.stack(sampled_segments)  # [B, T, C, H, W]
    label_indices = [phase2idx[label] for label in labels]
    return segment_tensor, torch.tensor(label_indices)


def select_collate(model_type, num_frames):
    """Return the correct collate function based on the model type."""
    if model_type == "cnn_lstm":
        return partial(collate_cnn_lstm, num_frames=num_frames)
    elif model_type == "cnn_tcn":
        return partial(collate_cnn_tcn, num_frames=num_frames)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
