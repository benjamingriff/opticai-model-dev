import os
import csv
import cv2
from torch.utils.data import Dataset


class Cataract21Dataset(Dataset):
    def __init__(self, data_root, split="train", transform=None):
        """
        Args:
            data_root: Path to the dataset root (e.g., ~/data/cataract)
            split: 'train', 'val', or 'test' — assumes CSV has a column or filter
            transform: Torchvision transform to apply to each frame or image
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.samples = self._load_annotations()

    def _load_annotations(self):
        annotations_path = os.path.join(
            self.data_root, "annotations", "cat21_annotations.csv"
        )
        samples = []
        with open(annotations_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["split"] == self.split:
                    video_path = os.path.join(self.data_root, "videos", row["video"])
                    label = row["phase"]
                    start_frame = int(row["start_frame"])
                    end_frame = int(row["end_frame"])
                    samples.append((video_path, label, start_frame, end_frame))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, start, end = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame from {video_path} at {start}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        if self.transform:
            frame = self.transform(frame)

        return frame, label
