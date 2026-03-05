import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from kpp.config import PredictorConfig

logger = logging.getLogger(__name__)


class PerformanceModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        hidden_size_2: int = 64,
        head_hidden_size: int = 32,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size_2),
            nn.GELU(),
        )

        # Head 1: Response Time Specialization
        self.head_rt = nn.Sequential(
            nn.Linear(hidden_size_2, head_hidden_size), nn.GELU(), nn.Linear(head_hidden_size, 1)
        )

        # Head 2: Throughput Specialization
        self.head_tp = nn.Sequential(
            nn.Linear(hidden_size_2, head_hidden_size), nn.GELU(), nn.Linear(head_hidden_size, 1)
        )

        # Head 3: CPU Usage Specialization
        self.head_cpu = nn.Sequential(
            nn.Linear(hidden_size_2, head_hidden_size), nn.GELU(), nn.Linear(head_hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_features = self.trunk(x)

        out_rt = self.head_rt(shared_features)
        out_tp = self.head_tp(shared_features)
        out_cpu = self.head_cpu(shared_features)

        # Concatenate along the feature dimension to return shape (batch, 3)
        return torch.cat([out_rt, out_tp, out_cpu], dim=1)


def train_model(
    config: PredictorConfig,
    service_name: str,
    model: PerformanceModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> float:
    """Trains a PerformanceModel, restores best weights in memory, and returns best test loss."""
    torch.manual_seed(42)
    best_test_loss = float("inf")
    best_state_dict = None

    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.training.weight_decay
    )
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
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)

            loss_rt = criterion(predictions[:, 0], batch_y[:, 0])
            loss_tp = criterion(predictions[:, 1], batch_y[:, 1])
            loss_cpu = criterion(predictions[:, 2], batch_y[:, 2])
            loss = loss_rt + loss_tp + loss_cpu
            loss.backward()
            optimizer.step()

            # loss = criterion(predictions, batch_y)
            # loss.backward()
            # optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_total_samples += batch_x.size(0)

        if train_total_samples == 0:
            raise RuntimeError(f"No training samples found for {service_name}.")
        train_loss /= train_total_samples

        model.eval()
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                test_total_samples += batch_x.size(0)

        if test_total_samples == 0:
            raise RuntimeError(f"No test samples found for {service_name}.")
        test_loss /= test_total_samples

        scheduler.step(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state_dict = copy.deepcopy(model.state_dict())

        train_rmse = np.sqrt(train_loss)
        test_rmse = np.sqrt(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    logger.info("Training complete. Best weights restored into model.")
    return best_test_loss


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    scaler: MinMaxScaler,
    target_columns: list[str],
    feature_names: list[str],
    x_feature_names: list[str] | None = None,
    log_transform_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Runs inference on the test set and inverts scaling.

    Returns (real_predictions, real_targets, user_counts_int) in original scale.

    Args:
        feature_names: Column names matching the scaler (all features, used for inverse transform).
        x_feature_names: Column names of the model input tensor X. Defaults to feature_names.
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

    x_names = x_feature_names if x_feature_names is not None else feature_names
    x_user_count_idx = x_names.index("User Count")
    feature_user_count_idx = feature_names.index("User Count")
    num_features = len(feature_names)

    all_predictions = []
    all_targets = []
    all_user_counts_norm = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            # Extract user count from flat (batch, features) input
            all_user_counts_norm.append(batch_x[:, x_user_count_idx].cpu().numpy())

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

    if log_transform_columns:
        for col in log_transform_columns:
            if col in feature_names:
                idx = feature_names.index(col)
                real_predictions[:, idx] = (10 ** real_predictions[:, idx]) - 1e-9
                real_targets[:, idx] = (10 ** real_targets[:, idx]) - 1e-9

                real_predictions[:, idx] = np.maximum(real_predictions[:, idx], 0.0)
                real_targets[:, idx] = np.maximum(real_targets[:, idx], 0.0)

    # Inverse-transform user counts via dummy array
    dummy_uc = np.zeros((len(user_counts_norm), num_features))
    dummy_uc[:, feature_user_count_idx] = user_counts_norm
    user_counts_int = (
        scaler.inverse_transform(dummy_uc)[:, feature_user_count_idx].round().astype(int)
    )

    return real_predictions, real_targets, user_counts_int
