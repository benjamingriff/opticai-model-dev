import sys
import os

# sys.path.append(os.path.abspath(".."))

from datasets import get_dataset
from dotenv import load_dotenv
from rich import print
from torchvision import transforms

load_dotenv()

data_path = os.getenv("DATA_PATH")

dataset_name = "cataract-21"

DatasetClass = get_dataset(dataset_name)
dataset = DatasetClass(
    data_root=os.path.join(data_path, dataset_name),
    mode="segment",
)

print(dataset.data_root)
print(dataset.mode)
print(dataset.transform)
print(dataset.samples[:10])
print(len(dataset))
print(dataset.get_labels())

tensor, label = dataset[0]
print(tensor.shape)
print(label)

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

DatasetClass = get_dataset(dataset_name)
dataset = DatasetClass(
    data_root=os.path.join(data_path, dataset_name),
    mode="segment",
    transform=transform,
)

tensor, label = dataset[0]
print(tensor.shape)
print(label)
