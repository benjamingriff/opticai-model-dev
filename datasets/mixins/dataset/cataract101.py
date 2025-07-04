import os
import csv
from collections import defaultdict
from labels.mappings.cataract101 import normalise_phase


class Cataract101Mixin:
    def _load_phase_map(self):
        phase_map = {}
        folder = os.path.join(self.data_root, "phases.csv")
        with open(folder, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                phase_map[row["Phase"]] = row["Meaning"]
        return phase_map

    def _load_video_info(self):
        folder = os.path.join(self.data_root, "videos.csv")
        with open(folder, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            return {row["VideoID"]: row for row in reader}

    def _load_labels_dict(self):
        labels_dict = defaultdict(list)
        phase_map = self._load_phase_map()
        folder = os.path.join(self.data_root, "annotations.csv")
        with open(folder, "r") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                video_id = row["VideoID"]
                frame = int(row["FrameNo"])
                phase = normalise_phase(phase_map[row["Phase"]])
                labels_dict[video_id].append((frame, phase))
        return labels_dict

    def _load_all_samples(self):
        samples = []
        labels_dict = self._load_labels_dict()
        for video_id, labels in labels_dict.items():
            labels = sorted(labels, key=lambda x: x[0])
            video_path = os.path.join(self.data_root, "videos", f"case_{video_id}.mp4")
            samples.extend(self._load_sequence_samples(labels, video_path))
        return samples

    def _load_single_video_samples(self, video_path):
        video_id = video_path.split("case_")[1].split(".mp4")[0]
        labels_dict = self._load_labels_dict()
        labels = labels_dict[video_id]
        return self._load_sequence_samples(labels, video_path)

    def _load_sequence_samples(self, labels, video_path):
        frame_list, label_list = [], []

        all_video_info = self._load_video_info()
        video_info = all_video_info[video_path.split("case_")[1].split(".mp4")[0]]
        total_frames = int(video_info["Frames"])

        first_frame = labels[0][0]
        if first_frame != 1:
            start_frame = 1
            frame_list.extend([i for i in range(start_frame, first_frame)])
            label_list.extend(["Idle" for _ in range(start_frame, first_frame)])

        for i in range(len(labels) - 1):
            start_frame = labels[i][0]
            end_frame = labels[i + 1][0]
            label = labels[i][1]
            frame_list.extend([i for i in range(start_frame, end_frame)])
            label_list.extend([label for _ in range(start_frame, end_frame)])

        last_frame = labels[-1][0]
        last_label = labels[-1][1]
        if last_frame != total_frames:
            frame_list.extend([i for i in range(last_frame, total_frames)])
            label_list.extend([last_label for _ in range(last_frame, total_frames)])

        return [(video_path, frame_list, label_list)]
