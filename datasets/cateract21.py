import os
import csv
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torch


class Cataract21Dataset(Dataset):
    def __init__(self, data_root, mode="frame", transform=None):
        self.data_root = os.path.expanduser(data_root)
        self.transform = transform
        self.mode = mode
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for csv_file in [f for f in os.listdir(self.data_root) if f.endswith(".csv")]:
            video_file = csv_file.replace(".csv", ".mp4")
            video_path = os.path.join(self.data_root, video_file)
            csv_path = os.path.join(self.data_root, csv_file)

            with open(csv_path, "r") as f:
                reader = csv.DictReader(f, fieldnames=["frame", "label"])
                if self.mode == "frame":
                    for row in reader:
                        frame = int(row["frame"])
                        label = row["label"]
                        samples.append((video_path, frame, label))
                elif self.mode == "segment":
                    prev_label = None
                    start = None
                    current_frame = None
                    rows = list(reader)
                    for i, row in enumerate(rows):
                        label = row["label"]
                        frame = int(row["frame"])
                        if label != prev_label:
                            if prev_label is not None:
                                samples.append(
                                    (video_path, start, current_frame - 1, prev_label)
                                )
                            start = frame
                        prev_label = label
                        current_frame = frame + 1
                    if prev_label is not None:
                        samples.append(
                            (video_path, start, current_frame - 1, prev_label)
                        )
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "frame":
            video_path, frame_idx, label = self.samples[idx]
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError(f"Failed to read frame {frame_idx}")
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.transform:
                image = self.transform(image)
            return image, label

        elif self.mode == "segment":
            video_path, start_frame, end_frame, label = self.samples[idx]
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frames = []
            for f in range(start_frame, end_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            cap.release()
            video_tensor = torch.stack(frames)
            return video_tensor, label
