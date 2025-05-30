import os
import csv
import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict


class Cataract101Dataset(Dataset):
    def __init__(self, data_root, mode="frame", transform=None):
        self.data_root = os.path.expanduser(data_root)
        self.mode = mode
        self.transform = transform

        self.video_dir = os.path.join(self.data_root, "videos")
        self.annotation_path = os.path.join(self.data_root, "annotations.csv")
        self.phase_path = os.path.join(self.data_root, "phases.csv")

        self.phase_map = self._load_phase_map()
        self.samples = self._load_samples()

    def _load_phase_map(self):
        phase_map = {}
        with open(self.phase_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                phase_map[row["Phase"]] = row["Meaning"]
        return phase_map

    def _load_samples(self):
        frame_labels = defaultdict(list)

        with open(self.annotation_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                video_id = row["VideoID"]
                frame = int(row["FrameNo"])
                phase = self.phase_map[row["Phase"]]
                frame_labels[video_id].append((frame, phase))

        samples = []

        for video_id, labels in frame_labels.items():
            video_filename = f"case_{video_id}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            if self.mode == "frame":
                for frame, label in labels:
                    samples.append((video_path, frame, label))

            elif self.mode == "segment":
                prev_label = None
                start = None
                current_frame = None
                for frame, label in labels:
                    if label != prev_label:
                        if prev_label is not None:
                            samples.append(
                                (video_path, start, current_frame - 1, prev_label)
                            )
                        start = frame
                    prev_label = label
                    current_frame = frame + 1
                if prev_label is not None:
                    samples.append((video_path, start, current_frame - 1, prev_label))

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
