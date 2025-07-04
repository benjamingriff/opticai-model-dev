class SingleVideoMixin:
    def _load_single_video_samples(self):
        raise NotImplementedError("Implement in dataset format mixin")

    def __init__(self, video_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self._load_single_video_samples(video_path)
