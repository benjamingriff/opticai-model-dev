import torch
from torch.utils.data import Dataset


class BaseFeatureSequenceDataset(Dataset):
    def __init__(self, features_root):
        self.features_root = features_root
        self.samples = self._load_samples()

    def _load_samples(self):
        raise NotImplementedError("Subclasses must implement _load_samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_path, label_path = self.samples[idx]

        features = torch.load(feature_path)
        labels = torch.load(label_path)

        return features, labels
