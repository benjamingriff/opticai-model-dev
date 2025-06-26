def downsample_sequences(dataset, stride=2):
    """
    Uniformly subsample frames for sequence mode datasets.

    Args:
        dataset: Your full ConcatDataset or Dataset instance (in sequence mode).
        stride: Downsampling factor (e.g. 2 means keep every 2nd frame).
    Returns:
        New dataset with updated frame_list and label_list.
    """
    assert hasattr(dataset, "datasets"), "Expected a ConcatDataset"

    for subdataset in dataset.datasets:
        new_samples = []
        for sample in subdataset.samples:
            video_path, frame_list, label_list = sample
            downsampled_frames = frame_list[::stride]
            downsampled_labels = label_list[::stride]
            new_samples.append((video_path, downsampled_frames, downsampled_labels))
        subdataset.samples = new_samples

    return dataset
