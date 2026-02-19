import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kpp.logging_config import setup_logging
from kpp.predictor.model import PerformancesGRU
from kpp.predictor.pipeline import PerformancesDataPipeline
from kpp.predictor.visualizer import evaluate_and_plot

logger = logging.getLogger("predictor")


def train_model(
    service_name: str,
    model: PerformancesGRU,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
) -> None:
    """Handles the training and validation loops."""
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"gru_{service_name}.pth"
    config_path = out_dir / f"config_{service_name}.json"

    best_test_loss = float("inf")

    if config_path.exists():
        with open(config_path, "r") as f:
            try:
                old_config = json.load(f)
                is_same_arch = (
                    old_config.get("num_layers") == model.num_layers
                    and old_config.get("hidden_size") == model.hidden_size
                    and old_config.get("input_size") == model.gru.input_size
                    and old_config.get("output_size") == model.fc.out_features
                )

                if is_same_arch:
                    best_test_loss = old_config.get("best_test_loss", float("inf"))
                    logger.info(
                        f"Found existing model for {service_name}. Historical best to beat: {best_test_loss:.6f}"
                    )
                else:
                    logger.warning(
                        f"Architecture change detected for {service_name}. Ignoring old checkpoints."
                    )

            except json.JSONDecodeError:
                logger.warning("Config file is corrupted. Starting fresh.")

    hyperparams = {
        "service": service_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "input_size": model.gru.input_size,
        "output_size": model.fc.out_features,
    }

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
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

        train_rmse = np.sqrt(train_loss)
        test_rmse = np.sqrt(test_loss)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), model_path)

            hyperparams["best_test_loss"] = best_test_loss
            with open(config_path, "w") as f:
                json.dump(hyperparams, f, indent=4)

        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
            )

    logger.info(f"Training complete. Best model weights are maintained at: {model_path}")


def main() -> None:
    setup_logging("predictor")

    csv_path = "dataset/performance_results_medium.csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV data file not found: '{csv_path}'. Place your collected data file at this path."
        )

    sequence_length = 5
    target_cols = [
        "Response Time (s)",
        "Throughput (req/s)",
        "CPU Usage",
    ]

    pipeline = PerformancesDataPipeline(sequence_length, target_cols)
    datasets = pipeline.run(csv_path, split_ratio=0.8)

    # Derive feature list from the pipeline's schema, excluding non-numeric identifier columns.
    all_features = [
        col
        for col in PerformancesDataPipeline.REQUIRED_COLUMNS
        if col not in ("Timestamp", "Service")
    ]

    for service_name, data_split in datasets.items():
        logger.info(f"--- Service: {service_name} ---")

        X_train, y_train = data_split["train"]
        X_test, y_test = data_split["test"]

        logger.info(f"Train Shape: {X_train.shape} (Samples, Window, Features)")
        logger.info(f"Test Shape:  {X_test.shape}")

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_size = X_train.shape[2]
        output_size = y_train.shape[1]

        model = PerformancesGRU(
            input_size=input_size,
            hidden_size=64,
            output_size=output_size,
            num_layers=2,
            dropout=0.1,
        )

        train_model(service_name, model, train_loader, test_loader, epochs=100)

        logger.info(f"Loading the best saved weights for {service_name}...")
        model_path = Path("models") / f"gru_{service_name}.pth"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. Training may not have saved a checkpoint "
                f"(no epoch improved on the historical best)."
            )
        model.load_state_dict(torch.load(model_path, weights_only=True))

        logger.info(f"Evaluating and plotting {service_name}...")

        service_scaler = pipeline.scalers.get(service_name)
        if service_scaler is None:
            logger.warning(f"No scaler found for {service_name}. Skipping evaluation.")
            continue

        evaluate_and_plot(
            model=model,
            test_loader=test_loader,
            scaler=service_scaler,
            target_columns=target_cols,
            service_name=service_name,
            feature_names=all_features,
        )


if __name__ == "__main__":
    main()
