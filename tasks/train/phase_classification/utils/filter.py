from torch.utils.data import Subset, ConcatDataset


def filter_dataset_by_labels(dataset, allowed_labels):
    assert hasattr(dataset, "datasets"), "Expected ConcatDataset"
    filtered_datasets = []

    for subdataset in dataset.datasets:
        indices = [
            i
            for i, sample in enumerate(subdataset.samples)
            if sample[3] in allowed_labels
        ]

        if indices:
            filtered_datasets.append(Subset(subdataset, indices))

    return ConcatDataset(filtered_datasets)
