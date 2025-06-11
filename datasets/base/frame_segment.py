import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor


class BaseFrameSegmentDataset(Dataset):
    def __init__(self, data_root, mode, transform):
        self.data_root = os.path.expanduser(data_root)
        self.mode = mode
        self.transform = transform
        self.samples = None

    def get_labels(self):
        return sorted(set(sample[-1] for sample in self.samples))

    def _load_samples(self):
        """
        This method should be implemented by subclasses to load dataset-specific samples.
        """
        raise NotImplementedError("Subclasses must implement _load_samples")

    def _load_frame(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        return image

    def _load_segment(self, video_path, start_frame, end_frame):
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

            if not isinstance(frame, torch.Tensor):
                frame = to_tensor(frame)

            frames.append(frame)

        cap.release()

        if not frames:
            raise RuntimeError(
                f"Failed to read frames {start_frame}-{end_frame} from {video_path}"
            )

        video_tensor = torch.stack(frames)
        return video_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.mode == "frame":
            video_path, frame_idx, label = self.samples[idx]
            image = self._load_frame(video_path, frame_idx)
            return image, label
        elif self.mode == "segment":
            video_path, start_frame, end_frame, label = self.samples[idx]
            video_tensor = self._load_segment(video_path, start_frame, end_frame)
            return video_tensor, label
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")
