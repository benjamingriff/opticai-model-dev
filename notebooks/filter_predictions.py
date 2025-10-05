import os, glob, json, math
import numpy as np
from labels.phases import idx2phase

# CONFIG
# Process all raw inference files; skip smoothed outputs. Raw files contain '.mp4.json'.
INPUT_GLOB = "outputs/infer/phase_segmentation/*.mp4.json"
OUT_DIR = "outputs/infer/phase_segmentation"
FPS = 25
STRIDE = 4
MAJ_WINDOWS = [9, 15, 21]  # big windows ok given downsample
MIN_SEG_SECONDS = [8, 10, 12]  # expected min phase durations
# END CONFIG

phase2idx = {name: i for i, name in idx2phase.items()}


def normalize_label_value(label_value):
    # Preserve original label structure, but help idx conversion
    if isinstance(label_value, list):
        if len(label_value) == 0:
            return None
        # Use the first entry if label is wrapped in a single-element list
        return label_value[0]
    return label_value


def to_idx(v):
    v = normalize_label_value(v)
    if v is None:
        return -1
    if isinstance(v, int):
        return v
    return phase2idx.get(str(v), -1)


def majority_filter(arr, k=5):
    if k <= 1:
        return arr
    pad = k // 2
    padded = np.pad(arr, (pad, pad), mode="edge")
    out = np.empty_like(arr)
    for i in range(len(arr)):
        window = padded[i : i + k]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def remove_short_runs(arr, min_len=5):
    if min_len <= 1:
        return arr
    arr = arr.copy()
    n = len(arr)
    start = 0
    while start < n:
        end = start
        while end + 1 < n and arr[end + 1] == arr[start]:
            end += 1
        run_len = end - start + 1
        if run_len < min_len:
            left_val = arr[start - 1] if start > 0 else None
            right_val = arr[end + 1] if end + 1 < n else None
            if left_val is not None and right_val is not None and left_val != right_val:
                l = start - 1
                while l - 1 >= 0 and arr[l - 1] == left_val:
                    l -= 1
                left_run = start - l
                r = end + 1
                while r + 1 < n and arr[r + 1] == right_val:
                    r += 1
                right_run = r - end
                fill_val = left_val if left_run >= right_run else right_val
            else:
                fill_val = left_val if left_val is not None else right_val
            if fill_val is not None:
                arr[start : end + 1] = fill_val
        start = end + 1
    return arr


def save_smoothed(base_json_path, sm_pred_idx, frame2phase, tag):
    T = len(frame2phase)
    out = {}
    for i in range(T):
        # Preserve original ground-truth label exactly as-is (can be list)
        original_label = frame2phase[str(i)].get("label", None)
        out[str(i)] = {
            "pred": idx2phase[int(sm_pred_idx[i])],
            "label": original_label,
        }
    base = os.path.splitext(os.path.basename(base_json_path))[0]
    out_path = os.path.join(OUT_DIR, f"{base}.{tag}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved", out_path)


def main():
    # Only process the base raw file; skip any smoothed outputs explicitly
    json_paths = [
        p
        for p in sorted(glob.glob(INPUT_GLOB))
        if ".smoothed." not in os.path.basename(p)
    ]
    if not json_paths:
        print("No JSONs found:", INPUT_GLOB)
        return

    for jp in json_paths:
        with open(jp) as f:
            frame2phase = json.load(f)
        T = len(frame2phase)
        preds = [frame2phase[str(i)]["pred"] for i in range(T)]
        pred_idx = np.array([to_idx(p) for p in preds], dtype=int)

        for sec in MIN_SEG_SECONDS:
            min_len = int(math.ceil((sec * FPS) / STRIDE))
            for k in MAJ_WINDOWS:
                sm = majority_filter(pred_idx, k=k)
                sm = remove_short_runs(sm, min_len=min_len)
                tag = f"smoothed.k{k}.min{min_len}"
                save_smoothed(jp, sm, frame2phase, tag)


if __name__ == "__main__":
    main()
