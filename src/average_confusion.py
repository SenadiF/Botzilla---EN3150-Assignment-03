
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from src.DepthwiseSeparableConv import DepthwiseSeparableConv

import src.model_light as model_light_module

model_light_module.DepthwiseSeparableConv = DepthwiseSeparableConv

from src.model_light import WasteImageNNLight
from src import test_loader

# Class order must match the label order used by your dataset.
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

N_RUNS = 5
NUM_CLASSES = len(CLASS_NAMES)

WEIGHTS_DIR = Path("models/experiments")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Generate an average confusion matrix over five runs."
    )
    parser.add_argument(
        "--model",
        choices=["A", "B"],
        required=True,
        help="Choose Model A or Model B.",
    )
    args = parser.parse_args()

    # Select the correct model class and saved-weight filename.
    if args.model == "A":
        model_class = WasteImageNN
        weights_pattern = "model_A_run{}.pth"
    else:
        model_class = WasteImageNNLight
        weights_pattern = "model_B_light_run{}.pth"

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Evaluating Model {args.model} on the test set...")

    all_confusion_matrices = []

    # Evaluate each of the five saved runs.
    for run in range(1, N_RUNS + 1):
        weights_path = WEIGHTS_DIR / weights_pattern.format(run)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Saved weights not found: {weights_path}\n"
                "Check that the five experiment weight files exist."
            )

        model = model_class()

        state_dict = torch.load(
            weights_path,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                y_pred.extend(predictions.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        )

        accuracy = np.trace(cm) / cm.sum() * 100
        print(f"Run {run}: accuracy = {accuracy:.2f}%")

        all_confusion_matrices.append(cm)

    # Shape: (5 runs, 9 true classes, 9 predicted classes).
    cms = np.asarray(all_confusion_matrices, dtype=float)

    # Average number of predictions in each cell.
    mean_counts = cms.mean(axis=0)

    # Normalize each run by its true-class row.
    row_totals = cms.sum(axis=2, keepdims=True)

    percentages = np.divide(
        cms,
        row_totals,
        out=np.zeros_like(cms),
        where=row_totals != 0,
    ) * 100

    mean_percentages = percentages.mean(axis=0)
    sd_percentages = percentages.std(axis=0, ddof=1)

    # Save raw matrices for later analysis.
    raw_output = OUTPUT_DIR / (
        f"Model{args.model}_confusion_all_runs.npy"
    )
    np.save(raw_output, cms)

    print("\nMean recall per class:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"{name}: "
            f"{mean_percentages[i, i]:.1f}% "
            f"+/- {sd_percentages[i, i]:.1f}%"
        )

    # Report the six largest average row-normalized confusions.
    print("\nLargest average confusions (true -> predicted):")
    confusions = [
        (mean_percentages[i, j], i, j)
        for i in range(NUM_CLASSES)
        for j in range(NUM_CLASSES)
        if i != j
    ]

    for value, i, j in sorted(confusions, reverse=True)[:6]:
        print(
            f"{CLASS_NAMES[i]} -> {CLASS_NAMES[j]}: "
            f"{value:.1f}% "
            f"(mean count: {mean_counts[i, j]:.1f})"
        )

    # Draw the two confusion-matrix panels.
    fig, axes = plt.subplots(
        1, 2, figsize=(19, 8), dpi=200
    )

    panels = [
        (
            axes[0],
            mean_counts,
            "Mean count",
            lambda i, j: f"{mean_counts[i, j]:.1f}",
        ),
        (
            axes[1],
            mean_percentages,
            "True-class percentage (mean +/- SD)",
            lambda i, j: (
                f"{mean_percentages[i, j]:.1f}\n"
                f"+/- {sd_percentages[i, j]:.1f}"
            ),
        ),
    ]

    for ax, data, title, cell_label in panels:
        image = ax.imshow(data, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        ax.set_title(
            f"Model {args.model}: {title}",
            fontsize=12,
        )
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(
            CLASS_NAMES, rotation=45, ha="right", fontsize=9
        )
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

        threshold = data.max() / 2

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax.text(
                    j,
                    i,
                    cell_label(i, j),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if data[i, j] > threshold
                        else "black"
                    ),
                )

    fig.tight_layout()

    output_path = OUTPUT_DIR / (
        f"Model{args.model}_confusion_avg.png"
    )
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved confusion matrix: {output_path}")
    print(f"Saved raw matrices: {raw_output}")


if __name__ == "__main__":
    main()
