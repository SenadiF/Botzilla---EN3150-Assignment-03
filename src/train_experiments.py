import random
import time
from pathlib import Path
from src.DepthwiseSeparableConv import DepthwiseSeparableConv
import src.model_light as model_light_module
import numpy as np
import pandas as pd
import torch

from src import WasteImageNN, WasteImageNNLight
from src import train_loader, val_loader, test_loader
from src.train import train
from src.evaluate_experiments import evaluate_experiment


model_light_module.DepthwiseSeparableConv = DepthwiseSeparableConv
def main():
    NUM_RUNS = 5
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3

    # Paths are relative to the project root.
    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models" / "experiments"
    results_dir = project_root / "results"

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
       # ("Model A", WasteImageNN, "model_A"),
        ("Model B", WasteImageNNLight, "model_B_light"),
    ]

    all_results = []

    for model_name, model_class, filename in experiments:
        print("\n" + "=" * 60)
        print(f"{model_name}: {NUM_RUNS} independent runs")
        print("=" * 60)

        for run in range(1, NUM_RUNS + 1):
            seed = 1000 + run

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            print(f"\n{model_name} - Run {run}/{NUM_RUNS}")

            # Create a fresh model for each run.
            model = model_class()

            start = time.perf_counter()

            train_losses, val_losses = train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_ephohs=NUM_EPOCHS,
                lr=LEARNING_RATE,
            )

            elapsed = time.perf_counter() - start

            metrics = evaluate_experiment(model, test_loader)

            all_results.append({
                "Model": model_name,
                "Run": run,
                "Seed": seed,
                "Accuracy": metrics["accuracy"],
                "Weighted Precision": metrics["precision"],
                "Weighted Recall": metrics["recall"],
                "Weighted F1": metrics["f1_score"],
                "Training Time (seconds)": elapsed,
            })

            # Keep every run's model weights.
            save_path = models_dir / f"{filename}_run{run}.pth"
            torch.save(model.state_dict(), save_path)

            # Save the confusion matrix for each run separately.
            cm_path = results_dir / f"{filename}_run{run}_cm.csv"
            np.savetxt(
                cm_path,
                metrics["confusion_matrix"],
                delimiter=",",
                fmt="%d",
            )

            print(f"Training time: {elapsed:.2f} seconds")
            print(f"Saved weights: {save_path}")

    # Save individual run results.
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        results_dir / "experiment_results.csv",
        index=False,
    )

    # Calculate mean and sample standard deviation across five runs.
    metric_columns = [
        "Accuracy",
        "Weighted Precision",
        "Weighted Recall",
        "Weighted F1",
        "Training Time (seconds)",
    ]

    summary_df = (
        results_df
        .groupby("Model")[metric_columns]
        .agg(["mean", "std"])
    )

    summary_df.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in summary_df.columns
    ]

    summary_df = summary_df.reset_index()

    summary_df.to_csv(
        results_dir / "experiment_summary.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("INDIVIDUAL RUN RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("MEAN AND STANDARD DEVIATION")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    print("\nAll experiment results have been saved.")


if __name__ == "__main__":
    main()
