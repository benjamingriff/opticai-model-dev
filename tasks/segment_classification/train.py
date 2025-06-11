import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from models.cnn_lstm import CNNLSTM
from datasets import get_dataset
from .utils.collate import segment_collate
from .utils.split import split_dataset_by_video
from rich import print

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def evaluate_model(model, val_loader, device, label_names, output_dir):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y, lengths in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, lengths)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
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


def train(cfg):
    output_dir = "outputs/segment_classification"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading datasets...")
    dataset = load_datasets(cfg)

    print("Splitting datasets...")
    train_set, val_set = split_dataset_by_video(dataset, cfg.get("val_split", 0.2))

    print("Loading data loaders...")
    train_loader = DataLoader(
        train_set,
        batch_size=cfg["batch_size"],
        shuffle=True,
        collate_fn=segment_collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg["batch_size"],
        shuffle=False,
        collate_fn=segment_collate,
    )

    num_classes = len(dataset.datasets[0].get_labels())

    print(f"Number of classes: {num_classes}")

    print("Loading model...")
    model = CNNLSTM(num_classes=num_classes)
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
        for i, (x, y, lengths) in enumerate(train_loader):
            print(f"Training batch {i}: {x.shape}, {y.shape}, {lengths}")
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad()
            logits = model(x, lengths)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        print("[bold yellow]Validating...[/bold yellow]")
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for i, (x, y, lengths) in enumerate(val_loader):
                print(f"Validating batch {i}: {x.shape}, {y.shape}, {lengths}")
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
    label_names = dataset.datasets[0].get_labels()
    cm_path, report_path = evaluate_model(
        model, val_loader, cfg["device"], label_names, output_dir
    )

    print(f"Best model saved to {os.path.join(output_dir, 'models/best.pt')}")
    print(f"Confusion matrix saved to {cm_path}")
    print(f"Classification report saved to {report_path}")
