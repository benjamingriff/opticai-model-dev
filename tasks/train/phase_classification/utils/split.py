import os
import random
from torch.utils.data import Subset


def split_dataset_by_video(dataset, val_split=0.2, seed=42):
    assert hasattr(dataset, "datasets"), "Expected a ConcatDataset"

    video_to_indices = {}
    global_index = 0

    for ds in dataset.datasets:
        if isinstance(ds, Subset):
            base_ds = ds.dataset
            for i in ds.indices:
                sample = base_ds.samples[i]
                video_name = os.path.basename(sample[0])
                video_to_indices.setdefault(video_name, []).append(global_index)
                global_index += 1
        else:
            for i, sample in enumerate(ds.samples):
                video_name = os.path.basename(sample[0])
                video_to_indices.setdefault(video_name, []).append(global_index)
                global_index += 1

    videos = list(video_to_indices.keys())
    random.seed(seed)
    random.shuffle(videos)
    n_val = int(len(videos) * val_split)
    val_videos = set(videos[:n_val])
    train_indices, val_indices = [], []

    for video, indices in video_to_indices.items():
        if video in val_videos:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
