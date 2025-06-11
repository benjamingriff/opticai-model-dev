import os
import csv
from datasets.base.frame_segment import BaseFrameSegmentDataset
from labels.mappings.cataract21 import normalize_phase


class Cataract21Dataset(BaseFrameSegmentDataset):
    def __init__(self, data_root, mode="frame", transform=None):
        super().__init__(data_root, mode=mode, transform=transform)
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        Load samples for the Cataract21 dataset and normalize phase labels.
        """
        samples = []
        for data_path in ["training", "validation"]:
            for csv_file in [
                f
                for f in os.listdir(os.path.join(self.data_root, data_path))
                if f.endswith(".csv")
            ]:
                video_file = csv_file.replace(".csv", ".mp4")
                video_path = os.path.join(self.data_root, data_path, video_file)
                csv_path = os.path.join(self.data_root, data_path, csv_file)

                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f, fieldnames=["frame", "label"])
                    if self.mode == "segment":
                        prev_label = None
                        start = None
                        current_frame = None
                        rows = list(reader)
                        for i, row in enumerate(rows):
                            label = row["label"]
                            normalized_label = normalize_phase(label)
                            frame = int(row["frame"])
                            if normalized_label != prev_label:
                                if prev_label is not None:
                                    samples.append(
                                        (
                                            video_path,
                                            start,
                                            current_frame - 1,
                                            prev_label,
                                        )
                                    )
                                start = frame
                            prev_label = normalized_label
                            current_frame = frame + 1
                        if prev_label is not None:
                            samples.append(
                                (
                                    video_path,
                                    start,
                                    current_frame - 1,
                                    prev_label,
                                )
                            )
        return samples
