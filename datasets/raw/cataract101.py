import os
import csv
from collections import defaultdict
from datasets.raw.base.frame_segment_sequence import BaseFrameSegmentSequenceDataset
from labels.mappings.cataract101 import normalise_phase


class Cataract101Dataset(BaseFrameSegmentSequenceDataset):
    def __init__(self, data_root, mode, transform=None):
        super().__init__(data_root, mode=mode, transform=transform)
        self.video_dir = os.path.join(self.data_root, "videos")
        self.annotation_path = os.path.join(self.data_root, "annotations.csv")
        self.phase_path = os.path.join(self.data_root, "phases.csv")
        self.video_info_path = os.path.join(self.data_root, "videos.csv")
        self.video_info = self._load_video_info()
        self.phase_map = self._load_phase_map()
        self.samples = self._load_samples()

    def _load_phase_map(self):
        phase_map = {}
        with open(self.phase_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                phase_map[row["Phase"]] = row["Meaning"]
        return phase_map

    def _load_video_info(self):
        with open(self.video_info_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            return {row["VideoID"]: row for row in reader}

    def _load_samples(self):
        frame_labels = defaultdict(list)

        with open(self.annotation_path, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                video_id = row["VideoID"]
                frame = int(row["FrameNo"])
                phase = normalise_phase(self.phase_map[row["Phase"]])
                frame_labels[video_id].append((frame, phase))

        samples = []
        for video_id, labels in frame_labels.items():
            labels = sorted(labels, key=lambda x: x[0])

            video_filename = f"case_{video_id}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            if self.mode == "frame":
                for i in range(len(labels) - 1):
                    start_frame = labels[i][0]
                    end_frame = labels[i + 1][0]
                    label = labels[i][1]

                    for frame in range(start_frame, end_frame):
                        samples.append((video_path, frame, label))

            elif self.mode == "segment":
                for i in range(len(labels) - 1):
                    start_frame = labels[i][0]
                    end_frame = labels[i + 1][0] - 1
                    label = labels[i][1]

                    samples.append((video_path, start_frame, end_frame, label))

                last_start_frame = labels[-1][0]
                last_label = labels[-1][1]
                max_frame = int(self.video_info[video_id]["Frames"])
                samples.append((video_path, last_start_frame, max_frame, last_label))

            elif self.mode == "sequence":
                frame_list = [frame for frame, _ in labels]
                label_list = [phase for _, phase in labels]
                samples.append((video_path, frame_list, label_list))
                breakpoint()

                # This needs to change because the data is nott correct. It needs to be more like == "frame"....

        return samples
