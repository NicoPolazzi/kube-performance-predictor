import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

logger = logging.getLogger("predictor.visualizer")


def evaluate_and_plot(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    target_columns: list[str],
    service_name: str,
    feature_names: list[str],
):
    """
    Runs inference on the test set, inverts the scaling, and plots mean ± std
    of predictions vs ground truth grouped by concurrent user count.

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

    if "User Count" not in feature_names:
        raise ValueError("'User Count' must be present in feature_names for x-axis grouping.")

    user_count_idx = feature_names.index("User Count")
    num_features = len(feature_names)

    all_predictions = []
    all_targets = []
    all_user_counts_norm = []

    model.eval()
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            predictions = model(batch_X)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            # Extract user count from the last timestep of each input window
            all_user_counts_norm.append(batch_X[:, -1, user_count_idx].cpu().numpy())

    pred_array = np.concatenate(all_predictions, axis=0)
    target_array = np.concatenate(all_targets, axis=0)
    user_counts_norm = np.concatenate(all_user_counts_norm, axis=0)

    # Inverse-transform predictions and targets
    target_indices = [feature_names.index(col) for col in target_columns]
    dummy_pred = np.zeros((len(pred_array), num_features))
    dummy_target = np.zeros((len(target_array), num_features))
    for i, idx in enumerate(target_indices):
        dummy_pred[:, idx] = pred_array[:, i]
        dummy_target[:, idx] = target_array[:, i]

    real_predictions = scaler.inverse_transform(dummy_pred)
    real_targets = scaler.inverse_transform(dummy_target)

    # Inverse-transform user counts via dummy array
    dummy_uc = np.zeros((len(user_counts_norm), num_features))
    dummy_uc[:, user_count_idx] = user_counts_norm
    user_counts_int = scaler.inverse_transform(dummy_uc)[:, user_count_idx].round().astype(int)

    unique_users = sorted(np.unique(user_counts_int))

    num_targets = len(target_columns)
    _, axes = plt.subplots(num_targets, 1, figsize=(12, 4 * num_targets), sharex=True)

    if num_targets == 1:
        axes = [axes]

    for i, col_name in enumerate(target_columns):
        ax = axes[i]
        target_idx = target_indices[i]

        mean_pred_list: list[float] = []
        std_pred_list: list[float] = []
        mean_true_list: list[float] = []
        std_true_list: list[float] = []

        for u in unique_users:
            mask = user_counts_int == u
            mean_pred_list.append(real_predictions[mask, target_idx].mean())
            std_pred_list.append(real_predictions[mask, target_idx].std())
            mean_true_list.append(real_targets[mask, target_idx].mean())
            std_true_list.append(real_targets[mask, target_idx].std())

        x = np.array(unique_users)
        mean_pred = np.array(mean_pred_list)
        std_pred = np.array(std_pred_list)
        mean_true = np.array(mean_true_list)
        std_true = np.array(std_true_list)

        ax.plot(x, mean_true, label="Ground Truth", color="blue", linewidth=2)
        ax.fill_between(x, mean_true - std_true, mean_true + std_true, color="blue", alpha=0.15)

        ax.plot(x, mean_pred, label="GRU Prediction", color="red", linestyle="--", linewidth=2)
        ax.fill_between(x, mean_pred - std_pred, mean_pred + std_pred, color="red", alpha=0.15)

        ax.set_title(f"{service_name} - {col_name}")
        ax.set_ylabel(col_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.xlabel("Concurrent Users")
    plt.tight_layout()

    output_dir = Path("plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{service_name}_predictions.png"
    plt.savefig(file_path)

    logger.info(f"Saved plot for {service_name} at {file_path}")
