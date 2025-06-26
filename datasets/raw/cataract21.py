import os
import csv
from datasets.raw.base.frame_segment_sequence import BaseFrameSegmentSequenceDataset
from labels.mappings.cataract21 import normalise_phase


class Cataract21Dataset(BaseFrameSegmentSequenceDataset):
    def __init__(self, data_root, mode, transform=None):
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
                    if self.mode == "frame":
                        for row in reader:
                            frame = int(row["frame"])
                            label = row["label"]
                            samples.append((video_path, frame, label))
                    elif self.mode == "segment":
                        prev_label = None
                        start = None
                        current_frame = None
                        rows = list(reader)
                        for i, row in enumerate(rows):
                            label = row["label"]
                            normalized_label = normalise_phase(label)
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

                    elif self.mode == "sequence":
                        frame_list = []
                        label_list = []
                        for row in reader:
                            frame = int(row["frame"])
                            label = normalise_phase(row["label"])
                            frame_list.append(frame)
                            label_list.append(label)
                        samples.append((video_path, frame_list, label_list))

                    else:
                        raise ValueError(f"Unsupported mode: {self.mode}")
        return samples
