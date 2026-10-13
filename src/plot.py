import matplotlib.pyplot as plt
import torch

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
