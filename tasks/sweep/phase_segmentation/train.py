import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from datasets import get_dataset
from models import get_model
from .utils.collate import select_collate
from .utils.split import split_dataset
from .utils.filter import downsample_sequences
from labels.phases import PHASES
from rich import print


def get_transform():
    return transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])


def load_datasets(cfg):
    datasets = []
    for dataset_name in cfg["datasets"]:
        DatasetClass = get_dataset(dataset_name, cfg["task"], cfg["scope"])
        dataset = DatasetClass(
            data_root=os.path.expanduser(os.path.join(cfg["data_path"], dataset_name)),
            transform=get_transform(),
        )
        datasets.append(dataset)
        print(f"{dataset_name} loaded")
    return ConcatDataset(datasets)


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def train(cfg):
    sweep_variants = cfg.get("sweep", None)

    if sweep_variants:
        for variant in sweep_variants:
            print(f"\n[yellow]Running variant: {variant['name']}[/yellow]")
            variant_cfg = cfg.copy()
            variant_cfg.update(variant)
            variant_cfg["output_dir"] = os.path.join(
                "outputs", cfg["task"], "sweep", variant["name"]
            )
            _train_single_model(variant_cfg)
    else:
        cfg["output_dir"] = os.path.join("outputs", cfg["task"], "default")
        _train_single_model(cfg)


def _train_single_model(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)
    print("Loading datasets...")
    dataset = load_datasets(cfg)

    stride = cfg["downsample_stride"]
    dataset = downsample_sequences(dataset, stride=stride)
    print(f"Applied temporal downsampling with stride: {stride}")

    train_set, val_set = split_dataset(dataset, cfg.get("val_split"))

    num_classes = len(PHASES)
    print(f"Number of classes: {num_classes}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=select_collate(cfg["model"]),
        num_workers=1,
        pin_memory=cfg["device"] == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=select_collate(cfg["model"]),
        num_workers=1,
        pin_memory=cfg["device"] == "cuda",
    )

    print("Loading model...")
    model = get_model(cfg["model"], num_classes=num_classes, **cfg["tcn"])
    model = model.to(cfg["device"])

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 3
    no_improve_epochs = 0

    print("Training...")
    for epoch in range(cfg["num_epochs"]):
        print(f"[yellow]Epoch {epoch + 1}[/yellow]")
        print(f"  Train dataset size: {len(train_set)} sequences")
        model.train()
        total_loss = 0

        for i, (x, y) in enumerate(train_loader):
            x = x.to(cfg["device"])
            y = y.to(cfg["device"])

            if epoch == 0 and i == 0:
                print(f"  [cyan]Batch input x shape: {x.shape}[/cyan]")
                print(f"  [cyan]Batch labels y shape: {y.shape}[/cyan]")

            optimizer.zero_grad()
            outputs = model(x)

            if epoch == 0 and i == 0:
                print(f"  [cyan]Model stage 1 output shape: {outputs[0].shape}[/cyan]")

            loss = 0
            for stage_idx, out in enumerate(outputs):
                out = out.permute(0, 2, 1).reshape(-1, num_classes)
                y_flat = y.reshape(-1)
                stage_loss = criterion(out, y_flat)
                loss += stage_loss
                if epoch == 0 and i == 0:
                    print(
                        f"  [cyan]Stage {stage_idx + 1} loss: {stage_loss.item():.4f}[/cyan]"
                    )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"  [green]Epoch {epoch + 1} average loss: {avg_loss:.4f}[/green]")

        # Validation
        model.eval()
        total_correct = 0
        total_frames = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(cfg["device"])
                y = y.to(cfg["device"])
                outputs = model(x)
                final_output = outputs[-1]
                preds = torch.argmax(final_output, dim=1)
                preds = preds.permute(0, 1)
                total_correct += (preds == y).sum().item()
                total_frames += y.numel()

        acc = total_correct / total_frames
        print(f"  [bold blue]Validation framewise accuracy: {acc:.4f}[/bold blue]")

        model_dir = os.path.join(cfg["output_dir"], "models")
        if acc > best_acc and abs(acc - best_acc) > 0.01:
            best_acc = acc
            no_improve_epochs = 0
            save_model(model, os.path.join(model_dir, "best.pt"))
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("[red]Early stopping![/red]")
            break
