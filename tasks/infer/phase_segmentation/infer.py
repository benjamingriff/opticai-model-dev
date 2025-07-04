import os
import json
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import get_dataset
from models import get_model
from labels.phases import idx2phase, PHASES
from .utils.filter import downsample_sequences
from rich import print


def get_transform():
    return transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


def infer(cfg, video_path):
    print("Starting inference...")

    output_dir = f"outputs/infer/{cfg['task']}"
    os.makedirs(output_dir, exist_ok=True)

    print("Initialising dataset...")
    dataset = get_dataset(cfg["dataset"], cfg["task"], cfg["scope"])(
        video_path=video_path, transform=get_transform()
    )

    print("Downsampling dataset...")
    dataset = downsample_sequences(dataset, stride=cfg["downsample_stride"])

    print("Initialising dataloader...")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    num_classes = len(PHASES)
    print(f"Number of classes: {num_classes}")

    print("Initialising model...")
    model = get_model(
        cfg["model"], num_classes=num_classes, precomputed_features=False, **cfg["tcn"]
    )
    model_path = cfg["checkpoint"]
    model.load_state_dict(torch.load(model_path, map_location=cfg["device"]))
    model = model.to(cfg["device"])
    model.eval()
    print(f"Loaded model from [green]{model_path}[/green]")

    print("Running inference...")
    video_tensor, label_list = next(
        iter(dataloader)
    )  # What should I do if the dataloader has multiple datasets / samples?
    video_tensor = video_tensor.to(cfg["device"])

    with torch.no_grad():
        outputs = model(video_tensor)
        final_output = outputs[-1]
        preds = torch.argmax(final_output, dim=1).squeeze(0)
        print("[bold green]Finished inference![/bold green]")

    print("Saving predictions...")
    frame2phase = {
        str(idx): {
            "pred": idx2phase[pred.item()],
            "label": label_list[idx] if label_list is not None else None,
        }
        for idx, pred in enumerate(preds)
    }

    output_path = os.path.join(output_dir, f"{os.path.basename(video_path)}.json")
    with open(output_path, "w") as f:
        json.dump(frame2phase, f, indent=2)

    print(f"Predictions saved to [green]{output_path}[/green]")
