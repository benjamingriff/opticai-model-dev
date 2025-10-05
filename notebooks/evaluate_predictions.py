import os, glob, json, csv
import numpy as np
from labels.phases import idx2phase, PHASES

INPUT_GLOB = "outputs/infer/phase_segmentation/*.json"
OUT_CSV = "outputs/infer/phase_segmentation/metrics.csv"

phase2idx = {name: i for i, name in idx2phase.items()}


def normalize_label_value(label_value):
    if isinstance(label_value, list):
        if len(label_value) == 0:
            return None
        return label_value[0]
    return label_value


def to_idx(v):
    v = normalize_label_value(v)
    if v is None:
        return -1
    if isinstance(v, int):
        return v
    return phase2idx.get(str(v), -1)


def compute_metrics(pred_idx, label_idx):
    mask = label_idx >= 0
    y_true = label_idx[mask]
    y_pred = pred_idx[mask]
    if len(y_true) == 0:
        return None, None
    acc = float((y_true == y_pred).mean())
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if f1s else None
    return acc, macro_f1


def main():
    rows = [("video", "variant", "frame_acc", "macro_f1", "n_labelled_frames")]
    for jp in sorted(glob.glob(INPUT_GLOB)):
        base = os.path.basename(jp)
        variant = "raw"
        if ".smoothed." in base:
            variant = base.split(".json")[0].split(".", 1)[
                1
            ]  # everything after first dot
        with open(jp) as f:
            frame2phase = json.load(f)
        T = len(frame2phase)
        preds = [frame2phase[str(i)]["pred"] for i in range(T)]
        labels_raw = [frame2phase[str(i)].get("label", None) for i in range(T)]
        pred_idx = np.array([to_idx(p) for p in preds], dtype=int)
        label_idx = np.array([to_idx(l) for l in labels_raw], dtype=int)
        acc, f1 = compute_metrics(pred_idx, label_idx)
        n_lbl = int(np.sum(label_idx >= 0))
        rows.append(
            (
                base,
                variant,
                None if acc is None else f"{acc:.4f}",
                None if f1 is None else f"{f1:.4f}",
                n_lbl,
            )
        )

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)
    print("Saved", OUT_CSV)


if __name__ == "__main__":
    main()
