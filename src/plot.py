
import matplotlib.pyplot as plt
import numpy as np

## Helper functions
def plot_training_loss(train_losses, validation_losses):
    """Plot training loss."""

    plt.figure(figsize=(10, 5))

    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs,train_losses, marker='o', label="Train_loss")
    plt.plot(epochs,validation_losses, marker='s', label="Validation loss")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_cm_marices(cm, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    # row-normalize: each row sums to 100% -> diagonal is per-class recall
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = 100 * cm / row_sums.clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    panels = [
        (cm, "Confusion Matrix (counts)", "d"),
        (cm_norm, "Confusion Matrix (% of true class)", ".1f"),
    ]

    for ax, (matrix, title, fmt) in zip(axes, panels):
        im = ax.imshow(matrix, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(title)

        # write the value in each cell, white text on dark cells
        threshold = matrix.max() / 2
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt),
                        ha="center", va="center", fontsize=9,
                        color="white" if matrix[i, j] > threshold else "black")

    plt.tight_layout()
    plt.show()

