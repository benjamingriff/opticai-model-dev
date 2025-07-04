class AllVideosMixin:
    def _load_all_samples(self):
        raise NotImplementedError("Implement in dataset format mixin")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.samples = self._load_all_samples()
