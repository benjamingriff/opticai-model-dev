import torch
import cv2
from torchvision.transforms.functional import to_tensor
from PIL import Image


class SequenceModeMixin:
    def __getitem__(self, idx):
        sample = self.samples[idx]

        if len(sample) == 3:
            video_path, frame_list, label_list = sample
        elif len(sample) == 2:
            video_path, frame_list = sample
            label_list = None
        else:
            raise ValueError(f"Invalid sample format: {sample}")

        cap = cv2.VideoCapture(video_path)
        frames = []
        for frame_idx in frame_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)

            if not isinstance(frame, torch.Tensor):
                frame = to_tensor(frame)
            frames.append(frame)
        cap.release()

        video_tensor = torch.stack(frames)  # (T, C, H, W)

        return video_tensor, label_list
