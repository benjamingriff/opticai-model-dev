import os
import csv
from labels.mappings.cataract21 import normalise_phase


class Cataract21Mixin:
    def _load_all_samples(self):
        samples = []
        for split in ["training", "validation"]:
            folder = os.path.join(self.data_root, split)
            for fname in os.listdir(folder):
                if fname.endswith(".csv"):
                    video = fname.replace(".csv", ".mp4")
                    video_path = os.path.join(folder, video)
                    csv_path = os.path.join(folder, fname)
                    samples.extend(self._load_sequence_samples(csv_path, video_path))
        return samples

    def _load_single_video_samples(self, video_path):
        csv_path = video_path.replace(".mp4", ".csv")
        return self._load_sequence_samples(csv_path, video_path)

    def _load_sequence_samples(self, csv_path, video_path):
        frame_list, label_list = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f, fieldnames=["frame", "label"]):
                frame_list.append(int(row["frame"]))
                label_list.append(normalise_phase(row["label"]))
        return [(video_path, frame_list, label_list)]
