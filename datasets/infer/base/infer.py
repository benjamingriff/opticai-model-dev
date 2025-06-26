import os
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import to_tensor


class BaseInferenceDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = os.path.expanduser(video_path)
        self.transform = transform
        self.frame_indices = self._get_all_frame_indices()

    def _get_all_frame_indices(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return list(range(total_frames))

    def _load_frame(self, frame_idx):
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(
                f"Failed to read frame {frame_idx} from {self.video_path}"
            )
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        if not isinstance(image, torch.Tensor):
            image = to_tensor(image)
        return image

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, _):
        frames = []
        for i in self.frame_indices:
            image = self._load_frame(i)
            frames.append(image)
        video_tensor = torch.stack(frames)  # (T, C, H, W)
        return video_tensor, torch.tensor(self.frame_indices)
