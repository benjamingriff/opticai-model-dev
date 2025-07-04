import os
import matplotlib.pyplot as plt
import random
import torch
from collections import defaultdict
from datasets import get_dataset
from torchvision import transforms
from rich.console import Console
from rich.table import Table

console = Console()


def run(cfg):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset_name = cfg["dataset"]
    DatasetClass = get_dataset(dataset_name, cfg["task"], cfg["scope"])
    dataset = DatasetClass(
        data_root=os.path.join(cfg["data_path"], dataset_name),
        mode=cfg["mode"],
        transform=transform,
    )

    console.print(f"Loaded {len(dataset)} samples")

    if cfg["mode"] == "frame":
        for i in random.sample(range(len(dataset)), 5):
            image, label = dataset[i]
            if isinstance(image, torch.Tensor) and image.ndim == 4:
                image = image[len(image) // 2]
            plt.imshow(image.permute(1, 2, 0))
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.show()

    elif cfg["mode"] == "segment":
        video_index = defaultdict(list)

        for sample in dataset.samples:
            video_path, start, end, label = sample
            video_index[video_path].append((start, end, label))

        video_name = random.choice(list(video_index.keys()))
        console.print(f"[bold cyan]Segments from video: {video_name}[/bold cyan]")

        table = Table()
        table.add_column("Label")
        table.add_column("Frames")
        table.add_column("Start")
        table.add_column("End")

        for start, end, label in video_index[video_name]:
            table.add_row(label, str(end - start + 1), str(start), str(end))

        console.print(table)
