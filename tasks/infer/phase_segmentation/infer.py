import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.infer.cataract21 import Cataract21InferenceDataset
from models import get_model
from labels.phases import idx2phase, PHASES
from .utils.filter import downsample_inference_dataset
from rich import print


def get_transform():
    return transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


def infer(cfg, video_path):
    print("Starting inference...")

    output_dir = "outputs/infer/phase_segmentation"
    os.makedirs(output_dir, exist_ok=True)

    dataset = Cataract21InferenceDataset(video_path, transform=get_transform())
    dataset = downsample_inference_dataset(dataset, stride=cfg["downsample_stride"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    num_classes = len(PHASES)
    print(f"Number of classes: {num_classes}")

    model = get_model(
        cfg["model"], num_classes=num_classes, precomputed_features=False, **cfg["tcn"]
    )
    model_path = cfg["checkpoint"]
    model.load_state_dict(torch.load(model_path, map_location=cfg["device"]))
    model = model.to(cfg["device"])
    model.eval()
    print(f"Loaded model from [green]{model_path}[/green]")

    video_tensor, frame_indices = next(iter(dataloader))  # each is (1, T, ...)
    video_tensor = video_tensor.to(cfg["device"])  # already (1, T, C, H, W)
    frame_indices = frame_indices.squeeze(0)  # (T,)

    with torch.no_grad():
        outputs = model(video_tensor)  # (1, num_classes, T)
        final_output = outputs[-1]
        preds = torch.argmax(final_output, dim=1).squeeze(0)
        print("[bold green]Finished inference![/bold green]")

    frame2phase = {
        str(idx.item()): idx2phase[pred.item()]
        for idx, pred in zip(frame_indices, preds)
    }

    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}.json")
    with open(output_path, "w") as f:
        json.dump(frame2phase, f, indent=2)

    print(f"Predictions saved to [green]{output_path}[/green]")
