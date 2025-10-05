import os, glob, json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from labels.phases import PHASES, idx2phase

INPUT_GLOB = "outputs/infer/phase_segmentation/*.json"  # or "*.json"
OUT_DIR = "outputs/infer/phase_segmentation/plots"
FPS = 25
STRIDE = 4
TICK_SEC = 30  # x-axis tick spacing in seconds

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


def timeline_plot(pred_idx, label_idx, out_path, fps=25, stride=4, tick_sec=30):
    T = len(pred_idx)
    data = np.vstack([pred_idx, label_idx])
    max_idx = len(PHASES)
    data_plot = data.copy()
    data_plot[data_plot < 0] = max_idx
    base_colors = plt.get_cmap("tab20", max_idx).colors
    colors = list(base_colors) + [(0.9, 0.9, 0.9)]
    cmap = mcolors.ListedColormap(colors)

    # compute ticks in downsampled frames
    step_frames = max(1, int(math.ceil((tick_sec * fps) / stride)))
    ticks = list(range(0, T, step_frames))
    tick_labels = [f"{int(round(t * stride / fps))}s" for t in ticks]

    plt.figure(figsize=(14, 3.2))
    plt.imshow(data_plot, aspect="auto", cmap=cmap, vmin=0, vmax=max_idx)
    plt.yticks([0, 1], ["pred", "label"])
    plt.xticks(ticks, tick_labels, rotation=0)
    plt.title("Phase timeline")
    # Draw vertical lines at phase changes in predictions
    changes = np.where(np.diff(pred_idx) != 0)[0] + 0
    for c in changes:
        plt.axvline(c, color="white", alpha=0.15, linewidth=0.7)
    # Legend
    handles = [plt.Line2D([0], [0], color=colors[i], lw=6) for i in range(len(PHASES))]
    plt.legend(
        handles,
        PHASES,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        fontsize=8,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    json_paths = sorted(glob.glob(INPUT_GLOB))
    if not json_paths:
        print("No JSONs found:", INPUT_GLOB)
        return

    for jp in json_paths:
        with open(jp) as f:
            frame2phase = json.load(f)
        T = len(frame2phase)
        preds = [frame2phase[str(i)]["pred"] for i in range(T)]
        labels_raw = [frame2phase[str(i)].get("label", None) for i in range(T)]
        pred_idx = np.array([to_idx(p) for p in preds], dtype=int)
        label_idx = np.array([to_idx(l) for l in labels_raw], dtype=int)
        base_name = os.path.basename(jp)
        base = os.path.splitext(base_name)[0]
        # Determine variant folder: 'raw' or the smoothed tag
        if ".smoothed." in base_name:
            variant = base_name.split(".json")[0].split(".", 1)[1]
        else:
            variant = "raw"
        out_dir_variant = os.path.join(OUT_DIR, variant)
        os.makedirs(out_dir_variant, exist_ok=True)
        out_path = os.path.join(out_dir_variant, f"{base}.png")
        timeline_plot(
            pred_idx, label_idx, out_path, fps=FPS, stride=STRIDE, tick_sec=TICK_SEC
        )
        print("Saved", out_path)


if __name__ == "__main__":
    main()
