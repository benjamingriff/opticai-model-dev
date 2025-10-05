import os
import json
from collections import defaultdict
from rich import print
from rich.table import Table
from rich.console import Console
from rich.panel import Panel

console = Console()


def extract_phase_sequence(prediction_data):
    """
    Extract the sequence of phases from prediction data.
    Returns a list of (phase, start_frame, end_frame) tuples.
    """
    frames = sorted([int(k) for k in prediction_data.keys()])
    sequence = []

    if not frames:
        return sequence

    current_phase = prediction_data[str(frames[0])]["pred"]
    start_frame = frames[0]

    for frame in frames[1:]:
        frame_phase = prediction_data[str(frame)]["pred"]
        if frame_phase != current_phase:
            # Phase changed, record the previous phase
            sequence.append((current_phase, start_frame, frame - 1))
            current_phase = frame_phase
            start_frame = frame

    # Add the last phase
    sequence.append((current_phase, start_frame, frames[-1]))

    return sequence


def extract_gt_sequence(prediction_data):
    """
    Extract the ground truth sequence from prediction data.
    Returns a list of (phase, start_frame, end_frame) tuples.
    """
    frames = sorted([int(k) for k in prediction_data.keys()])
    sequence = []

    if not frames:
        return sequence

    current_phase = prediction_data[str(frames[0])]["label"][0]  # GT is in a list
    start_frame = frames[0]

    for frame in frames[1:]:
        frame_phase = prediction_data[str(frame)]["label"][0]
        if frame_phase != current_phase:
            # Phase changed, record the previous phase
            sequence.append((current_phase, start_frame, frame - 1))
            current_phase = frame_phase
            start_frame = frame

    # Add the last phase
    sequence.append((current_phase, start_frame, frames[-1]))

    return sequence


def create_video_comparison_table(video_name, pred_sequence, gt_sequence):
    """
    Create a table comparing prediction vs ground truth for a single video.
    Each row represents one phase in the sequence.
    """
    # Pad the shorter sequence with None
    max_length = max(len(pred_sequence), len(gt_sequence))
    pred_sequence_padded = pred_sequence + [(None, None, None)] * (
        max_length - len(pred_sequence)
    )
    gt_sequence_padded = gt_sequence + [(None, None, None)] * (
        max_length - len(gt_sequence)
    )

    # Create table
    table = Table(title=f"Video: {video_name}")
    table.add_column("Position", style="cyan", no_wrap=True)
    table.add_column("Prediction", style="red")
    table.add_column("Pred Frames", style="red", no_wrap=True)
    table.add_column("Ground Truth", style="green")
    table.add_column("GT Frames", style="green", no_wrap=True)
    table.add_column("Match", style="yellow")

    for i in range(max_length):
        pred_phase, pred_start, pred_end = pred_sequence_padded[i]
        gt_phase, gt_start, gt_end = gt_sequence_padded[i]

        # Determine if they match
        if pred_phase is None or gt_phase is None:
            match = "N/A"
            match_style = "dim"
        elif pred_phase == gt_phase:
            match = "✓"
            match_style = "green"
        else:
            match = "✗"
            match_style = "red"

        # Format phases for display
        pred_display = pred_phase if pred_phase is not None else "None"
        gt_display = gt_phase if gt_phase is not None else "None"

        # Format frame ranges
        pred_frames = (
            f"{pred_start}-{pred_end}"
            if pred_start is not None and pred_end is not None
            else "None"
        )
        gt_frames = (
            f"{gt_start}-{gt_end}"
            if gt_start is not None and gt_end is not None
            else "None"
        )

        table.add_row(
            str(i + 1),
            pred_display,
            pred_frames,
            gt_display,
            gt_frames,
            f"[{match_style}]{match}[/{match_style}]",
        )

    return table


def main():
    predictions_dir = "outputs/infer/phase_segmentation"

    # Collect all JSON files
    json_files = [f for f in os.listdir(predictions_dir) if f.endswith(".json")]

    print(f"[bold blue]Analyzing {len(json_files)} prediction files...[/bold blue]\n")

    for json_file in sorted(json_files):
        video_name = json_file.replace(".json", "")
        json_path = os.path.join(predictions_dir, json_file)

        # Load prediction data
        with open(json_path, "r") as f:
            prediction_data = json.load(f)

        # Extract sequences
        pred_sequence = extract_phase_sequence(prediction_data)
        gt_sequence = extract_gt_sequence(prediction_data)

        # Create and display table for this video
        table = create_video_comparison_table(video_name, pred_sequence, gt_sequence)
        console.print(table)
        console.print()  # Add spacing between tables


if __name__ == "__main__":
    main()
