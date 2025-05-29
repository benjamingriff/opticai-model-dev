import os
import argparse
import yaml
from datasets.cataract21 import Cataract21Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()
data_path = os.getenv("DATA_PATH")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    dataset = Cataract21Dataset(
        data_root=cfg["data_path"], split=cfg.get("split", "train"), transform=transform
    )

    print(f"Loaded {len(dataset)} samples")

    for i in range(5):
        image, label = dataset[i]
        plt.imshow(image.permute(1, 2, 0))
        plt.title(f"Label: {label}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
