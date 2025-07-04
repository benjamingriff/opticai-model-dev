from datasets.mixins.scope.single_video import SingleVideoMixin
from datasets.mixins.task.sequence import SequenceModeMixin
from datasets.mixins.dataset.cataract101 import Cataract101Mixin
from datasets.base.base import BaseDataset


class Cataract101PhaseSegmentationDatasetSingle(
    Cataract101Mixin, SequenceModeMixin, SingleVideoMixin, BaseDataset
):
    def __init__(self, video_path, transform=None):
        """
        Args:
            file_path (str): Path to the video file.
            transform (callable, optional): Transformations to apply to each frame.
        """
        self.video_path = video_path
        self.transform = transform
        super().__init__(video_path=video_path, transform=transform)
