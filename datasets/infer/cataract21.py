from datasets.infer.base.infer import BaseInferenceDataset


class Cataract21InferenceDataset(BaseInferenceDataset):
    def __init__(self, video_path, transform=None):
        super().__init__(video_path, transform)
