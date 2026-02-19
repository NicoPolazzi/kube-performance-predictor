from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader


def evaluate_and_plot(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    target_columns: list[str],
    service_name: str,
    feature_names: list[str],
):
    """
    Runs inference on the test set, inverts the scaling, and plots the results.

    For the inverse transformation we need also the user count, so we need to handle this
    discrepancy. In the predictions, user count is missing.
    """
    missing_in_features = [col for col in target_columns if col not in feature_names]
    if missing_in_features:
        raise ValueError(
            f"Target columns not found in feature_names: {missing_in_features}. "
            f"Available features: {feature_names}"
        )

    if len(feature_names) != scaler.n_features_in_:
        raise ValueError(
            f"feature_names has {len(feature_names)} columns but scaler expects "
            f"{scaler.n_features_in_}. Ensure feature_names matches the columns used during fitting."
        )

    all_predictions = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    pred_array = np.concatenate(all_predictions, axis=0)
    target_array = np.concatenate(all_targets, axis=0)

    num_features = len(feature_names)
    target_indices = [feature_names.index(col) for col in target_columns]
    dummy_pred = np.zeros((len(pred_array), num_features))
    dummy_target = np.zeros((len(target_array), num_features))
    for i, idx in enumerate(target_indices):
        dummy_pred[:, idx] = pred_array[:, i]
        dummy_target[:, idx] = target_array[:, i]

    real_predictions = scaler.inverse_transform(dummy_pred)
    real_targets = scaler.inverse_transform(dummy_target)

    num_targets = len(target_columns)
    _, axes = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets), sharex=True)

    if num_targets == 1:
        axes = [axes]  # Ensure it's iterable if only 1 target

    # Cap at 200 points for readability; dense plots with many samples become illegible.
    plot_limit = min(200, len(real_predictions))

    for i, col_name in enumerate(target_columns):
        ax = axes[i]
        target_idx = target_indices[i]
        y_true = real_targets[:plot_limit, target_idx]
        y_pred = real_predictions[:plot_limit, target_idx]

        ax.plot(y_true, label="Ground Truth", color="blue", linewidth=2)
        ax.plot(y_pred, label="GRU Prediction", color="red", linestyle="--", linewidth=2)

        ax.set_title(f"{service_name} - {col_name}")
        ax.set_ylabel(col_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel("Time Steps (Minutes)")
    plt.tight_layout()

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{service_name}_predictions.png"
    plt.savefig(file_path)

    print(f"Saved plot for {service_name} at {file_path}")
