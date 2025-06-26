import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset
from datasets import get_dataset
from models import get_model
from .utils.collate import select_collate
from .utils.split import split_dataset_by_video
from .utils.filter import filter_dataset_by_labels
from labels.phases import idx2phases
from rich import print

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from memory_profiler import profile


def get_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


def load_datasets(cfg):
    datasets = []
    for dataset_name in cfg["datasets"]:
        DatasetClass = get_dataset(dataset_name)
        dataset = DatasetClass(
            data_root=os.path.join(cfg["data_path"], dataset_name),
            mode="segment",
            transform=get_transform(),
        )
        datasets.append(dataset)

    return ConcatDataset(datasets)


def get_labels_from_subset(subset):
    ds = subset.dataset
    indices = subset.indices
    return sorted(set(ds.samples[i][3] for i in indices))


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def evaluate_model(model, val_loader, output_dir, cfg):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            if cfg["model"] == "cnn_lstm":
                x, y, lengths = batch
            else:
                x, y = batch
                lengths = None
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    labels = unique_labels(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=labels)
    label_names = [idx2phases[i] for i in labels]
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=label_names)
    report_path = os.path.join(output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    return cm_path, report_path


@profile
def train(cfg):
    output_dir = "outputs/phase_classification"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    dataset = load_datasets(cfg)

    print("Filtering datasets...")
    dataset = filter_dataset_by_labels(dataset, cfg["include_labels"])

    print("Splitting datasets...")
    train_set, val_set = split_dataset_by_video(dataset, cfg.get("val_split"))

    num_classes = 0
    for i, ds in enumerate(dataset.datasets):
        if isinstance(ds, Subset):
            labels = get_labels_from_subset(ds)
        else:
            labels = ds.get_labels()
        num_classes = max(num_classes, len(labels))

    print(f"Number of classes: {num_classes}")

    print("Loading data loaders...")

    pin_memory = cfg["device"] == "cuda"

    collate_fn = select_collate(cfg["model"], cfg.get("num_frames", 200))

    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=pin_memory,
    )

    print("Loading model...")
    model = get_model(cfg["model"], num_classes=num_classes)
    model = model.to(cfg["device"])

    print("Loading optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience = 3
    no_improve_epochs = 0

    print("Training...")
    for epoch in range(cfg["num_epochs"]):
        print(
            f"[bold yellow]Starting epoch {epoch + 1} of {cfg['num_epochs']}[/bold yellow]"
        )
        model.train()
        for i, batch in enumerate(train_loader):
            # print(f"Training batch {i}: {x.shape}, {y.shape}, {lengths}")
            if cfg["model"] == "cnn_lstm":
                x, y, lengths = batch
            else:
                x, y = batch
                lengths = None
            start = time.time()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            end = time.time()
            print(f"Batch {i} took {end - start:.2f}s")

        print("[bold yellow]Validating...[/bold yellow]")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # print(f"Validating batch {i}: {x.shape}, {y.shape}, {lengths}")
                if cfg["model"] == "cnn_lstm":
                    x, y, lengths = batch
                else:
                    x, y = batch
                    lengths = None
                x, y = x.to(cfg["device"]), y.to(cfg["device"])
                logits = model(x, lengths)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch + 1} complete — Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            no_improve_epochs = 0
            save_model(model, os.path.join(output_dir, "models/best.pt"))
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    print("Evaluating best model...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "models/best.pt")))
    cm_path, report_path = evaluate_model(model, val_loader, output_dir, cfg)

    print(f"Best model saved to {os.path.join(output_dir, 'models/best.pt')}")
    print(f"Confusion matrix saved to {cm_path}")
    print(f"Classification report saved to {report_path}")
