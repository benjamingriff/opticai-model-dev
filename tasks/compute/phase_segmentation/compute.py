import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from datasets import get_dataset
from torchvision.models import resnet18, ResNet18_Weights
from .utils.filter import downsample_sequences

from rich import print


def get_transform():
    return transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


def load_datasets(cfg):
    datasets = []
    for dataset_name in cfg["datasets"]:
        DatasetClass = get_dataset(dataset_name)
        dataset = DatasetClass(
            data_root=os.path.join(cfg["data_path"], dataset_name),
            mode="sequence",
            transform=get_transform(),
        )
        datasets.append(dataset)
    return ConcatDataset(datasets)


def compute_features(cfg):
    output_dir = os.path.join("features", cfg["task"])
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    dataset = load_datasets(cfg)

    stride = cfg["downsample_stride"]
    dataset = downsample_sequences(dataset, stride=stride)
    print(f"Applied temporal downsampling with stride: {stride}")

    dataloader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0
    )

    print(f"Loaded dataset with {len(dataset)} sequences")

    device = cfg["device"]

    resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
    resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
    resnet.to(device)
    resnet.eval()

    for idx, (frames, labels) in enumerate(dataloader):
        if idx % 10 == 0:
            print(f"Processing sample {idx} of {len(dataloader)}")

        frames = frames.squeeze(0).to(device)
        labels = labels[0]

        features = []
        with torch.no_grad():
            for frame in frames:
                frame = frame.unsqueeze(0)
                feat = resnet(frame).squeeze()
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze()
                features.append(feat.cpu())

        features_tensor = torch.stack(features)

        feature_path = os.path.join(output_dir, f"sample_{idx:04d}.pt")
        torch.save({"features": features_tensor, "labels": labels}, feature_path)
        print(f"Saved features: {feature_path}")

    print("Finished feature extraction!")
