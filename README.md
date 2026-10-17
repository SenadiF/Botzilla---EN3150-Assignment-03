##  EN3150 Assignment 03 Simple convolutional neural network to perform classification.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![R-CNN](https://img.shields.io/badge/CNN-Object%20Classification-8A2BE2?style=for-the-badge)

### Group Botzilla


dataset : https://archive.ics.uci.edu/dataset/908/realwaste



Run the project 

```
uv venv --python 3.12.0
```
```
uv sync
```

Download the dataset (extracts to `data/RealWasteData/`)

```
cd data
uv run python setup_dataset.py
cd ..
```

Train and evaluate Model A and Model B (run from the project root)

```
uv run python -m src.train
```

### Project Structure

```
Group_Project/
├── data/
│   ├── setup_dataset.py          # downloads and extracts the RealWaste dataset
│   ├── realwaste_data.csv        # image manifest: file path, class name, label
│   └── RealWasteData/            # dataset images (not tracked, created by setup_dataset.py)
├── src/
│   ├── __init__.py               # package exports
│   ├── dataset.py                # RealWasteDataset (custom PyTorch Dataset)
│   ├── dataloader.py             # stratified 70/15/15 split, transforms, DataLoaders
│   ├── model.py                  # Model A: standard CNN (WasteImageNN)
│   ├── DepthwiseSeparableConv.py # depthwise separable convolution block
│   ├── model_light.py            # Model B: depthwise separable CNN (WasteImageNNLight)
│   ├── train.py                  # trains Model A and Model B, saves weights and figures
│   ├── evaluate.py               # test accuracy, confusion matrix, precision, recall
│   ├── plot.py                   # loss curves and confusion matrix plots
│   └── inferance.py              # inference
├── notebooks/
│   ├── classification_models.ipynb  # Model A and Model B development
│   └── experiments.ipynb            # experiments
├── models/                       # saved model weights (.pth)
├── figs/                         # loss curves and confusion matrices
├── docs/
│   ├── final_report_latex/       # final report (LaTeX source and PDF)
│   ├── task/                     # assignment brief
│   └── setup_latex_on_vscode.md
├── main.py
├── pyproject.toml                # project dependencies (uv)
├── uv.lock
└── req.txt
```

### Branches

| Branch | Contents |
|---|---|
| `main` | Final code, models, figures and report |
| `experiments` | Development of Model A and Model B, training scripts and figures |
| `experiments_1` | SOTA fine-tuning and evaluation experiments, plus repeated runs of Model A and Model B |

The `experiments_1` branch includes:

- **SOTA fine-tuning and evaluation**: MobileNetV2 (`notebooks/experiments1.ipynb`) and EfficientNet-B0 (`notebooks/experiments2.ipynb`) fine-tuned on RealWaste, with the trained weights in `models/`
- **Optimizer selection and tuning**: Adam, SGD and SGD with momentum at learning rates 0.01, 0.001 and 0.0001 (`notebooks/experiments3.ipynb`)
- **Model B training and evaluation**: `notebooks/Model_B_Training_Evaluation.ipynb`
- **Repeated runs**: Model A and Model B trained 5 times each, with average accuracy and per-run and averaged confusion matrices (`src/train_experiments.py`, `src/evaluate_run.py`, `src/average_confusion.py`, results in `results/`)
