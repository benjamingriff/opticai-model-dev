import os
import argparse
import yaml
from dotenv import load_dotenv
from string import Template
import importlib

load_dotenv()


def load_config(path):
    with open(path) as f:
        raw = f.read()
        interpolated = Template(raw).substitute(os.environ)
        return yaml.safe_load(interpolated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    task = cfg["task"]
    module = importlib.import_module(f"tasks.{task}.run")
    run_fn = getattr(module, "run")
    run_fn(cfg)


if __name__ == "__main__":
    main()
