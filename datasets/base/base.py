from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        return f"<{self.__class__.__name__}: {len(self.samples)} samples>"
