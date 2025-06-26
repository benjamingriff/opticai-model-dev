def downsample_inference_dataset(dataset, stride=2):
    dataset.frame_indices = dataset.frame_indices[::stride]
    return dataset
