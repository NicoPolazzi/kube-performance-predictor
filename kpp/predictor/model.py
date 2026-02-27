import json
import logging
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from kpp.config import PredictorConfig

logger = logging.getLogger("predictor")


class PerformanceModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) — flatten full window
        return cast(torch.Tensor, self.net(x.reshape(x.size(0), -1)))


def train_model(
    config: PredictorConfig,
    service_name: str,
    model: PerformanceModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> None:
    """Trains a PerformanceModel and saves best weights."""
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{service_name}.pth"
    config_path = out_dir / f"config_{service_name}.json"

    best_test_loss = float("inf")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.scheduler.factor,
        patience=config.scheduler.patience,
        min_lr=config.scheduler.min_lr,
    )

    for epoch in range(epochs):
        train_loss = 0.0
        test_loss = 0.0
        train_total_samples = 0
        test_total_samples = 0

        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            train_total_samples += batch_X.size(0)

        if train_total_samples == 0:
            raise RuntimeError(f"No training samples found for {service_name}.")
        train_loss /= train_total_samples

        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                test_total_samples += batch_X.size(0)

        if test_total_samples == 0:
            raise RuntimeError(f"No test samples found for {service_name}.")
        test_loss /= test_total_samples

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_path)
            with open(config_path, "w") as f:
                json.dump(
                    {
                        "service": service_name,
                        "hidden_size": config.model.hidden_size,
                        "best_test_loss": best_test_loss,
                    },
                    f,
                    indent=4,
                )

        train_rmse = np.sqrt(train_loss)
        test_rmse = np.sqrt(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
            )

    logger.info(f"Training complete. Best model weights saved to: {model_path}")


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    target_columns: list[str],
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on the test set and inverts scaling.

    Returns (real_predictions, real_targets, user_counts_int) in original scale.
    Pure computation — no I/O or plotting.
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

    return real_predictions, real_targets, user_counts_int
