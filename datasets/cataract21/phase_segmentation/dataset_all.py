from datasets.mixins.scope.all_videos import AllVideosMixin
from datasets.mixins.task.sequence import SequenceModeMixin
from datasets.mixins.dataset.cataract21 import Cataract21Mixin
from datasets.base.base import BaseDataset


class Cataract21PhaseSegmentationDatasetAll(
    Cataract21Mixin, SequenceModeMixin, AllVideosMixin, BaseDataset
):
    def __init__(self, data_root, transform=None):
        """
        Args:
            data_root (str): Root directory of the Cataract21 dataset.
            transform (callable, optional): Transformations to apply to each frame.
        """
        self.data_root = data_root
        self.transform = transform
        super().__init__(data_root=data_root, transform=transform)
