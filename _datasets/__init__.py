from _datasets.raw.cataract21 import Cataract21Dataset
from _datasets.raw.cataract101 import Cataract101Dataset
from _datasets.infer.cataract21 import Cataract21InferenceDataset

DATASET_REGISTRY = {
    "cataract-21": Cataract21Dataset,
    "cataract-101": Cataract101Dataset,
    "cataract-21-infer": Cataract21InferenceDataset,
}


def get_dataset(name):
    return DATASET_REGISTRY[name]
