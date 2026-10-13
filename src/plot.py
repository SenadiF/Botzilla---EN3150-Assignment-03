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

# evaluvate test 
def evaluate_test(model, test_loader):
    """Evaluate model on test set."""
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    test_accuracy = 100 * correct / total
    print(f"\n{'='*50}")
    print(f"TEST ACCURACY: {test_accuracy:.2f}%")
    print(f"{'='*50}")

    return test_accuracy