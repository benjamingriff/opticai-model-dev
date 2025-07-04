import sys
import os

sys.path.append(os.path.abspath(".."))

from datasets import get_dataset
from dotenv import load_dotenv
from rich import print
from torchvision import transforms

load_dotenv()

data_path = os.getenv("DATA_PATH")

dataset_name = "cataract-21"
task = "phase_segmentation"
scope = "all"

dataset_path = os.path.expanduser(os.path.join(data_path, dataset_name))

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

DatasetClass = get_dataset(dataset_name, task, scope)
dataset = DatasetClass(
    data_root=dataset_path,
    transform=transform,
)

print(dataset.data_root)
print(dataset.transform)
print(dataset.samples[0])
print(len(dataset))

tensor, label = dataset[0]
print(tensor.shape)
print(label)


dataset_name = "cataract-21"
task = "phase_segmentation"
scope = "single"

video_path = os.path.join(dataset_path, "training/case_10.mp4")

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

DatasetClass = get_dataset(dataset_name, task, scope)
dataset = DatasetClass(
    video_path=video_path,
    transform=transform,
)

print(dataset.video_path)
print(dataset.transform)
print(dataset.samples)
print(len(dataset))
