import os
import shutil
import json
import uuid
from datetime import datetime
from annotations.base import AnnotationExporter


class SuperviselyExporter(AnnotationExporter):
    def __init__(self, labeler_login="system"):
        self.labeler_login = labeler_login

        self.phase_colors = {
            "Incision": "#FF6B6B",  # Red
            "Viscous agent injection": "#4ECDC4",  # Teal
            "Rhexis": "#45B7D1",  # Blue
            "Hydrodissection": "#96CEB4",  # Green
            "Phacoemulsification": "#FFEAA7",  # Yellow
            "Irrigation and aspiration": "#DDA0DD",  # Plum
            "Capsule polishing": "#98D8C8",  # Mint
            "Lens implant setting-up": "#F7DC6F",  # Gold
            "Viscous agent removal": "#BB8FCE",  # Purple
            "Tonifying and antibiotics": "#85C1E9",  # Light Blue
            "Idle": "#BDC3C7",  # Gray
        }

    def create_meta_json(self, output_dir, labels):
        """
        Create the meta.json file with tag definitions.

        Args:
            output_dir: Output directory path
            labels: List of unique label names
        """
        meta = {
            "classes": [],
            "tags": [],
            "projectType": "videos",
            "projectSettings": {
                "multiView": {
                    "enabled": False,
                    "tagName": None,
                    "tagId": None,
                    "isSynced": False,
                }
            },
        }

        for i, label in enumerate(sorted(set(labels))):
            tag = {
                "name": label,
                "value_type": "none",
                "color": self.phase_colors.get(
                    label, "#02C2FF"
                ),  # Use phase-specific color or default
                "id": 32862770 + i,  # Generate unique IDs
                "hotkey": "",
                "applicable_type": "imagesOnly",
                "classes": [],
                "target_type": "framesOnly",
            }
            meta["tags"].append(tag)

        meta_path = os.path.join(output_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)

    def create_video_annotation(
        self, video_path, frame_list, label_list, video_key=None
    ):
        """
        Create annotation JSON for a single video.

        Args:
            video_path: Path to the video file
            frame_list: List of frame indices
            label_list: List of labels corresponding to frames
            video_key: Optional custom key for the video

        Returns:
            dict: Annotation data structure
        """
        if video_key is None:
            video_key = str(uuid.uuid4()).replace("-", "")

        # Group consecutive frames with same labels into frame ranges
        frame_ranges = []
        current_start = frame_list[0]
        current_label = label_list[0]

        for i in range(1, len(frame_list)):
            if label_list[i] != current_label:
                # End current range and start new one
                frame_ranges.append(
                    {
                        "start_frame": current_start,
                        "end_frame": frame_list[i - 1],
                        "label": current_label,
                    }
                )
                current_start = frame_list[i]
                current_label = label_list[i]

        # Add the last range
        frame_ranges.append(
            {
                "start_frame": current_start,
                "end_frame": frame_list[-1],
                "label": current_label,
            }
        )

        # Create tags from frame ranges
        tags = []
        for range_info in frame_ranges:
            tag = {
                "name": range_info["label"],
                "labelerLogin": self.labeler_login,
                "updatedAt": datetime.utcnow().isoformat() + "Z",
                "createdAt": datetime.utcnow().isoformat() + "Z",
                "frameRange": [range_info["start_frame"], range_info["end_frame"]],
                "isFinished": True,
                "nonFinalValue": False,
                "key": str(uuid.uuid4()).replace("-", ""),
            }
            tags.append(tag)

        annotation = {
            "size": {
                "height": 540,
                "width": 720,
            },
            "description": "",
            "key": video_key,
            "tags": tags,
            "objects": [],
            "frames": [],
            "framesCount": max(frame_list) + 1 if frame_list else 0,
        }

        return annotation

    def export_dataset(self, dataset, output_dir, dataset_name):
        """
        Convert prelabelled training data into Supervisely video tag format.
        """

        annotation_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "ann"), exist_ok=True)

        # Collect all unique labels for meta.json
        all_labels = []

        for item in dataset.samples:
            video_path, frame_list, label_list = item
            all_labels.extend(label_list)

            shutil.copy(
                video_path,
                os.path.join(annotation_dir, "videos", os.path.basename(video_path)),
            )

            annotation = self.create_video_annotation(
                video_path, frame_list, label_list
            )

            # Save annotation file
            ann_path = os.path.join(
                annotation_dir, "ann", os.path.basename(video_path) + ".json"
            )
            with open(ann_path, "w") as f:
                json.dump(annotation, f, indent=4)

        self.create_meta_json(output_dir, all_labels)

    def create_combined_video_annotation(
        self,
        video_path,
        gt_frame_list,
        gt_label_list,
        pred_frame_list,
        pred_label_list,
        video_key=None,
    ):
        """
        Create annotation JSON for a single video with both GT and predictions.

        Args:
            video_path: Path to the video file
            gt_frame_list: Ground truth frame indices
            gt_label_list: Ground truth labels
            pred_frame_list: Prediction frame indices
            pred_label_list: Prediction labels
            video_key: Optional custom key for the video

        Returns:
            dict: Annotation data structure with both GT and predictions
        """
        if video_key is None:
            video_key = str(uuid.uuid4()).replace("-", "")

        tags = []

        # Add ground truth tags
        if gt_frame_list and gt_label_list:
            gt_ranges = self._create_frame_ranges(gt_frame_list, gt_label_list)
            for range_info in gt_ranges:
                tag = {
                    "name": f"GT_{range_info['label']}",  # Prefix to distinguish
                    "labelerLogin": "ground_truth",  # Different labeler
                    "updatedAt": datetime.utcnow().isoformat() + "Z",
                    "createdAt": datetime.utcnow().isoformat() + "Z",
                    "frameRange": [range_info["start_frame"], range_info["end_frame"]],
                    "isFinished": True,
                    "nonFinalValue": False,
                    "key": str(uuid.uuid4()).replace("-", ""),
                }
                tags.append(tag)

        # Add prediction tags
        if pred_frame_list and pred_label_list:
            pred_ranges = self._create_frame_ranges(pred_frame_list, pred_label_list)
            for range_info in pred_ranges:
                tag = {
                    "name": f"PRED_{range_info['label']}",  # Prefix to distinguish
                    "labelerLogin": "model_prediction",  # Different labeler
                    "updatedAt": datetime.utcnow().isoformat() + "Z",
                    "createdAt": datetime.utcnow().isoformat() + "Z",
                    "frameRange": [range_info["start_frame"], range_info["end_frame"]],
                    "isFinished": True,
                    "nonFinalValue": False,
                    "key": str(uuid.uuid4()).replace("-", ""),
                }
                tags.append(tag)

        annotation = {
            "size": {
                "height": 540,
                "width": 720,
            },
            "description": "",
            "key": video_key,
            "tags": tags,  # Combined tags from both GT and predictions
            "objects": [],
            "frames": [],
            "framesCount": max(max(gt_frame_list or [0]), max(pred_frame_list or [0]))
            + 1,
        }

        return annotation

    def _create_frame_ranges(self, frame_list, label_list):
        """Helper method to create frame ranges from frame and label lists."""
        if not frame_list or not label_list:
            return []

        frame_ranges = []
        current_start = frame_list[0]
        current_label = label_list[0]

        for i in range(1, len(frame_list)):
            if label_list[i] != current_label:
                frame_ranges.append(
                    {
                        "start_frame": current_start,
                        "end_frame": frame_list[i - 1],
                        "label": current_label,
                    }
                )
                current_start = frame_list[i]
                current_label = label_list[i]

        # Add the last range
        frame_ranges.append(
            {
                "start_frame": current_start,
                "end_frame": frame_list[-1],
                "label": current_label,
            }
        )

        return frame_ranges

    def load_predictions(self, predictions_dir):
        """
        Load prediction JSON files and convert to dataset-like format.

        Args:
            predictions_dir: Directory containing prediction JSON files

        Returns:
            Dict mapping video_name to (frame_list, prediction_list)
        """
        predictions = {}

        for json_file in os.listdir(predictions_dir):
            if not json_file.endswith(".json"):
                continue

            video_name = json_file.replace(".json", "")
            json_path = os.path.join(predictions_dir, json_file)

            with open(json_path, "r") as f:
                frame2phase = json.load(f)

            # Convert to frame_list, prediction_list format
            frame_list = []
            prediction_list = []

            for frame_idx_str, data in frame2phase.items():
                frame_list.append(int(frame_idx_str))
                prediction_list.append(data["pred"])

            predictions[video_name] = (frame_list, prediction_list)

        return predictions

    def export_combined_annotations(
        self, dataset, predictions, output_dir, dataset_name
    ):
        """
        Export both ground truth and predictions in the same annotation files.
        """
        annotation_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "videos"), exist_ok=True)
        os.makedirs(os.path.join(annotation_dir, "ann"), exist_ok=True)

        # Collect all unique labels for meta.json
        all_labels = []

        for item in dataset.samples:
            video_path, gt_frame_list, gt_label_list = item
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Get predictions for this video
            pred_frame_list, pred_label_list = predictions.get(
                f"{video_name}.mp4", ([], [])
            )
            # Add labels to collection
            all_labels.extend([f"GT_{label}" for label in gt_label_list])
            all_labels.extend([f"PRED_{pred}" for pred in pred_label_list])

            # Copy video file
            shutil.copy(
                video_path,
                os.path.join(annotation_dir, "videos", os.path.basename(video_path)),
            )

            # Create combined annotation
            annotation = self.create_combined_video_annotation(
                video_path,
                gt_frame_list,
                gt_label_list,
                pred_frame_list,
                pred_label_list,
            )

            # Save annotation file
            ann_path = os.path.join(
                annotation_dir, "ann", os.path.basename(video_path) + ".json"
            )
            with open(ann_path, "w") as f:
                json.dump(annotation, f, indent=4)

        # Create meta.json with both GT and prediction labels
        self.create_meta_json(output_dir, all_labels)

    def export_predictions(self, predictions, output_dir, dataset_name):
        """
        Export model predictions to Supervisely format with distinct styling.
        """
        pass
