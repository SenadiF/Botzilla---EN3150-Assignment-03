import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_experiment(model, test_loader):
    """Evaluate a model and return its metrics."""

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = model.to(device)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)

            outputs = model(batch_x)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(batch_y.tolist())

    accuracy = 100 * accuracy_score(all_labels, all_predictions)

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_predictions)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-score: {f1:.4f}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
    }