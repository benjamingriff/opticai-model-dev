from datasets.cateract21 import Cataract21Dataset
from datasets.cateract101 import Cataract101Dataset

DATASET_REGISTRY = {
    "cataract21": Cataract21Dataset,
    "cataract101": Cataract101Dataset,
}


def get_dataset(name):
    return DATASET_REGISTRY[name]
