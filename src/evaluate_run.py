
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src import WasteImageNN, WasteImageNNLight
from src.dataloader import test_loader

# Fix the DepthwiseSeparableConv import issue in Model B
from src.DepthwiseSeparableConv import DepthwiseSeparableConv
import importlib

model_light_module = importlib.import_module("src.model_light")
model_light_module.DepthwiseSeparableConv = DepthwiseSeparableConv

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["A", "B"], required=True)
parser.add_argument("--run", type=int, required=True)
args = parser.parse_args()

# Select the model class and saved weights
if args.model == "A":
    model_class = WasteImageNN
    weights = Path(f"models/experiments/model_A_run{args.run}.pth")
else:
    model_class = WasteImageNNLight
    weights = Path(f"models/experiments/model_B_light_run{args.run}.pth")

if not weights.exists():
    raise FileNotFoundError(f"Model weights not found: {weights}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = model_class()
state_dict = torch.load(weights, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# Predict using the existing test set
y_true = []
y_pred = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        outputs = model(x)
        predictions = outputs.argmax(dim=1)

        y_pred.extend(predictions.cpu().numpy())
        y_true.extend(y.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate overall metrics
accuracy = accuracy_score(y_true, y_pred)

print(
    f"\nModel {args.model}, Run {args.run}: "
    f"Test accuracy = {accuracy * 100:.2f}%\n"
)

# Class-wise precision, recall and F1
p, r, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=range(len(CLASS_NAMES)),
    zero_division=0,
)

pm, rm, fm, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

pw, rw, fw, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print("Class-wise results:")
print("Class & Precision & Recall & F1-score \\\\")

for name, precision, recall, score in zip(CLASS_NAMES, p, r, f1):
    print(f"{name} & {precision:.4f} & {recall:.4f} & {score:.4f} \\\\")

print("\\hline")
print(f"Macro Average & {pm:.4f} & {rm:.4f} & {fm:.4f} \\\\")
print(f"Weighted Average & {pw:.4f} & {rw:.4f} & {fw:.4f} \\\\")

# Create the confusion matrix and row-normalized percentages
cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))

row_totals = cm.sum(axis=1, keepdims=True)
cm_pct = np.divide(
    cm * 100.0,
    row_totals,
    out=np.zeros_like(cm, dtype=float),
    where=row_totals != 0,
)

print("\nCorrect predictions per class:")
for i, name in enumerate(CLASS_NAMES):
    print(
        f"{name}: {cm[i, i]}/{cm[i].sum()} "
        f"(recall {cm_pct[i, i]:.1f}%)"
    )

# Plot counts and percentages side by side
fig, axes = plt.subplots(1, 2, figsize=(18, 8), dpi=200)

for ax, data, title, fmt in [
    (axes[0], cm, "Counts", "d"),
    (axes[1], cm_pct, "Percentage of true class", ".1f"),
]:
    im = ax.imshow(data, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title(f"Model {args.model}, Run {args.run}: {title}")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    threshold = data.max() / 2 if data.size else 0

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            value = format(data[i, j], fmt)
            ax.text(
                j, i, value,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i, j] > threshold else "black",
            )

fig.tight_layout()

# Save the image in the results folder
output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"Model{args.model}_confusion_run{args.run}.png"
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)

print(f"\nConfusion matrix saved to: {output_path}")
