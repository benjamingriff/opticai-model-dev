import os
import yaml
from dotenv import load_dotenv
from string import Template
import importlib

load_dotenv()

CONFIG_PATH = "./configs/train/phase_classification/cataract_cnn_lstm.yaml"


def load_config(path):
    with open(path) as f:
        raw = f.read()
        interpolated = Template(raw).substitute(os.environ)
        return yaml.safe_load(interpolated)


def main():
    cfg = load_config(CONFIG_PATH)

    task = cfg["task"]
    module = importlib.import_module(f"tasks.train.{task}.train")
    run_fn = getattr(module, "train")
    run_fn(cfg)


if __name__ == "__main__":
    main()
