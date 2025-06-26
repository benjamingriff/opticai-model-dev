import os
from datasets.features.base.features import BaseFeatureSequenceDataset


class Cataract21FeatureDataset(BaseFeatureSequenceDataset):
    def _load_samples(self):
        samples = []
        feature_dir = os.path.join(self.features_root, "cataract21")
        for filename in os.listdir(feature_dir):
            if filename.endswith(".pt") and filename.startswith("case_"):
                feature_path = os.path.join(feature_dir, filename)
                label_path = feature_path.replace(".pt", "_labels.pt")
                samples.append((feature_path, label_path))
        return samples
