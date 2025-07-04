def downsample_sequences(dataset, stride=2):
    """
    Uniformly subsample frames for sequence mode datasets.

    Args:
        dataset: A single Dataset instance or a ConcatDataset (in sequence mode).
        stride: Downsampling factor (e.g. 2 means keep every 2nd frame).
    Returns:
        The dataset with updated frame_list and label_list (if labels exist).
    """
    if hasattr(dataset, "datasets"):
        for subdataset in dataset.datasets:
            _downsample_single_dataset(subdataset, stride)
    else:
        _downsample_single_dataset(dataset, stride)

    return dataset


def _downsample_single_dataset(dataset, stride):
    """
    Helper function to downsample a single dataset.

    Args:
        dataset: A single Dataset instance.
        stride: Downsampling factor.
    """
    new_samples = []
    for sample in dataset.samples:
        if len(sample) == 3:
            video_path, frame_list, label_list = sample
            downsampled_frames = frame_list[::stride]
            downsampled_labels = label_list[::stride]
            new_samples.append((video_path, downsampled_frames, downsampled_labels))
        elif len(sample) == 2:
            video_path, frame_list = sample
            downsampled_frames = frame_list[::stride]
            new_samples.append((video_path, downsampled_frames))
        else:
            raise ValueError(f"Invalid sample format: {sample}")
    dataset.samples = new_samples
