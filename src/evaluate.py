import torch
from src import plot_cm_marices
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score
)

# evaluvate test
def evaluate_test(model, test_loader, cm_title=None, cm_save_path=None):
    """Evaluate model on test set. Saves the confusion matrix to cm_save_path if given."""
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    model.eval()
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            # Move data to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    # Accuracy 
    test_accuracy = 100 * correct / total
    print(f"\n{'='*50}")
    print(f"TEST ACCURACY: {test_accuracy:.4f}%")
    print(f"{'='*50}")

    # confusion matrix 
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\n{'='*50}")
    print(f"CM : {cm}")
    print(f"{'='*50}")
    # test_loader.dataset is a Subset -> .dataset is the RealWasteDataset
    plot_cm_marices(cm, class_names=test_loader.dataset.dataset.classes,
                    title=cm_title, save_path=cm_save_path)

    # Overall Precision
    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted"
    )
    print(f"\n{'='*50}")
    print(f"Precision: {precision:.4f}%")
    print(f"{'='*50}")
    
    # Overall Recall
    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted"
    )
    print(f"\n{'='*50}")
    print(f"Recall: {recall:.4f}%")
    print(f"{'='*50}")

    return test_accuracy, cm, precision, recall