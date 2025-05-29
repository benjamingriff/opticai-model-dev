# Optic AI Model Development

## 👁️ Surgical Phase Classification and Analysis

This repository is a modular and extensible framework for training, evaluating, and deploying deep learning models for cataract surgery video analysis.

It is designed to support:

* Surgical phase classification
* Temporal segmentation and boundary detection
* Surgical tool detection (future extension)
* Local experimentation with PyTorch (Apple M-series GPU-compatible)
* Cloud-scale training using AWS SageMaker or EC2 (Inferentia/Trn1 support)

---

## 📁 Project Structure

```
cataract-ai/
├── configs/                    # YAML config files for different tasks
│   ├── phase_classification.yaml
│   ├── tool_detection.yaml
│   └── ...
├── datasets/                  # Dataset loading and preprocessing modules
│   ├── cataract21.py
│   ├── cataract101.py
│   └── transforms.py
├── models/                    # Modular architectures
│   ├── cnn_lstm.py
│   ├── r3d.py
│   ├── swin_transformer.py
│   └── yolov5_tool_detection.py
├── tasks/                     # Training/eval logic per task
│   ├── phase_classification/
│   ├── phase_segmentation/
│   └── tool_detection/
├── scripts/                   # CLI entrypoints for training/inference
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── extract_segments.py
├── notebooks/                # Quick experiments & visualization
├── outputs/                  # Model outputs, logs, checkpoints
├── aws/                      # SageMaker/EC2 configs and scripts
├── requirements.txt
├── README.md
└── .env                      # Local data paths (optional)
```

---

## 🧠 Training Tasks

### Phase Classification (Default Task)

```bash
python scripts/train_model.py --config configs/phase_classification.yaml
```

### Tool Detection (Future)

```bash
python scripts/train_model.py --config configs/tool_detection.yaml
```

Each `config.yaml` file defines:

```yaml
task: phase_classification
model: cnn_lstm
num_epochs: 30
batch_size: 16
data_path: /Users/you/data/cataract/
```

Training logic is dispatched dynamically via `importlib`, making `scripts/train_model.py` universal.

---

## 🧰 Data Handling

* Keep large video datasets **outside the repo**, e.g., `~/data/cataract/`
* Use `.env` or config paths to point to your local or S3 data
* Scripts support extracting segments from long videos using `ffmpeg` or `opencv`

---

## 🚀 AWS Support

### A. Running on **SageMaker**

Use `aws/sagemaker_train.py` (to be added) or define via Python SDK:

```python
estimator = PyTorch(
    entry_point="scripts/train_model.py",
    source_dir=".",
    role=your_iam_role,
    instance_type="ml.g5.2xlarge",
    hyperparameters={
        "config": "/opt/ml/input/data/config/phase_classification.yaml"
    },
    dependencies=["requirements.txt"]
)

estimator.fit({
    "config": "s3://bucket/configs/",
    "training": "s3://bucket/cataract-videos/"
})
```

### B. Running on **EC2 / Inferentia**

* Use `scp` or `git clone` to deploy repo
* Install deps and run `python scripts/train_model.py` as usual
* Use Neuron SDK if targeting `trn1` or `inf2`

---

## ✅ Adding a New Model

Add your model class to models/ (e.g., models/my_model.py):

```python
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        ...
```

Register it in models/__init__.py:

```python
from models.my_model import MyModel
MODEL_REGISTRY = {
    ...
    "my_model": MyModel
}
```

Reference it in your config:

```yml
model: my_model
```

Call it with `get_model(cfg["model"], **kwargs)` in your task module

---

## ✅ Adding a New Task

1. Create a subfolder in `tasks/` (e.g., `tool_detection/`)
2. Add `train.py`, `eval.py`, etc.
3. Ensure `train(config)` exists
4. Create a YAML config in `configs/`
5. Run via:

```bash
python scripts/train_model.py --config configs/tool_detection.yaml
```

---

## ✅ Future Improvements

* Add Docker container for training/inference
* Add experiment tracking via W\&B or MLflow
* Support automatic hyperparameter tuning (e.g., Optuna or SageMaker HPO)
* Add CLI tool to upload/download data from S3

---

## 📦 Datasets Used

* [Cataract-21 Dataset](https://ftp.itec.aau.at/datasets/ovid/cat-21/)
* [Cataract-101 Dataset](https://ftp.itec.aau.at/datasets/ovid/cat-101/)
* [CATARACTS Dataset](https://ieee-dataport.org/open-access/cataracts) (for external validation)

---

## 🪣 Environment

Steps to setup a environment using UV.

Instal UV
```bash
pip install --upgrade uv
```

Create a virtual environment using UV with a specific python version.
```bash
uv venv --python python3.12 .venv
```

Activate the environment.
```bash
source .venv/bin/activate
```

or on windows.

```bash
.venv\Scripts\activate
```

Start installing the packages you need for dev.
```bash
uv pip install jupyter
```

Once you are happy with your requirements.
```bash
uv pip freeze > requirements.txt
```
### Revisiting the project

Install the lovingly prepared requirements.txt
```bash
uv pip install requirements.txt
```