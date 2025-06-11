from datasets.cataract21 import Cataract21Dataset
from datasets.cataract101 import Cataract101Dataset

DATASET_REGISTRY = {
    "cataract-21": Cataract21Dataset,
    "cataract-101": Cataract101Dataset,
}


def get_dataset(name):
    return DATASET_REGISTRY[name]
