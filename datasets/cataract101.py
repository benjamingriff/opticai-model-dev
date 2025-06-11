import os
import csv
from collections import defaultdict
from datasets.base.frame_segment import BaseFrameSegmentDataset


class Cataract101Dataset(BaseFrameSegmentDataset):
    def __init__(self, data_root, mode="frame", transform=None):
        super().__init__(data_root, mode=mode, transform=transform)
        self.video_dir = os.path.join(self.data_root, "videos")
        self.annotation_path = os.path.join(self.data_root, "annotations.csv")
        self.phase_path = os.path.join(self.data_root, "phases.csv")
        self.phase_map = self._load_phase_map()
        self.samples = self._load_samples()

    def _load_phase_map(self):
        phase_map = {}
        with open(self.phase_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                phase_map[row["Phase"]] = row["Meaning"]
        return phase_map

    def _load_samples(self):
        frame_labels = defaultdict(list)

        with open(self.annotation_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                video_id = row["VideoID"]
                frame = int(row["FrameNo"])
                phase = self.phase_map[row["Phase"]]
                frame_labels[video_id].append((frame, phase))

        samples = []
        for video_id, labels in frame_labels.items():
            # Sort labels by frame number
            labels = sorted(labels, key=lambda x: x[0])

            video_filename = f"case_{video_id}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            if self.mode == "frame":
                # Generate samples for each frame based on the label transitions
                for i in range(len(labels) - 1):
                    start_frame = labels[i][0]
                    end_frame = labels[i + 1][0]
                    label = labels[i][1]

                    for frame in range(start_frame, end_frame):
                        samples.append((video_path, frame, label))

            elif self.mode == "segment":
                # Generate samples for each segment based on the label transitions
                for i in range(len(labels) - 1):
                    start_frame = labels[i][0]
                    end_frame = labels[i + 1][0] - 1  # End frame is inclusive
                    label = labels[i][1]

                    samples.append((video_path, start_frame, end_frame, label))

                # Handle the last segment
                last_start_frame = labels[-1][0]
                last_label = labels[-1][1]
                # Assuming the last segment ends at the last frame of the video
                # You may need to replace `max_frame` with the actual last frame of the video
                max_frame = last_start_frame + 1000  # Example: Add a buffer
                samples.append((video_path, last_start_frame, max_frame, last_label))

        return samples
