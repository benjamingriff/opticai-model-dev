import os
from datasets import get_dataset
from annotations.factory import get_exporter


def load_dataset(cfg):
    DatasetClass = get_dataset(cfg["dataset"], cfg["task"], cfg["scope"])
    dataset = DatasetClass(
        data_root=os.path.expanduser(os.path.join(cfg["data_path"], cfg["dataset"])),
    )
    return dataset


def export(cfg):
    dataset = load_dataset(cfg)
    exporter = get_exporter(cfg["backend"])
    predictions = exporter.load_predictions(cfg["predictions_dir"])
    exporter.export_combined_annotations(
        dataset, predictions, cfg["output_dir"], cfg["dataset"]
    )
