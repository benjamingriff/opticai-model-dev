import os
import cv2
from torch.utils.data import Dataset
from PIL import Image


class BaseToolDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = os.path.expanduser(data_root)
        self.transform = transform
        self.samples = None

    def _load_samples(self):
        """
        This method should be implemented by subclasses to load dataset-specific samples.
        """
        raise NotImplementedError("Subclasses must implement _load_samples")

    def _load_tool_annotations(self):
        """
        Load tool annotations (e.g., polygons, bounding boxes).
        """
        raise NotImplementedError("Subclasses must implement _load_tool_annotations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, frame_idx, tool_label = self.samples[idx]
        image = self._load_frame(video_path, frame_idx)
        return image, tool_label
