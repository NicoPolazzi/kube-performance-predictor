import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kpp.config import PredictorConfig
from kpp.logging_config import setup_logging
from kpp.predictor.model import PerformanceModel
from kpp.predictor.pipeline import PerformanceDataPipeline
from kpp.predictor.visualizer import evaluate_and_plot

logger = logging.getLogger("predictor")


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


def main() -> None:
    setup_logging("predictor")
    config = PredictorConfig.from_yaml()

    csv_path = "dataset/performance_results_medium.csv"
    if not Path(csv_path).exists():
        raise FileNotFoundError(
            f"CSV data file not found: '{csv_path}'. Place your collected data file at this path."
        )

    target_cols = [
        "Response Time (s)",
        "Throughput (req/s)",
        "CPU Usage",
    ]

    pipeline = PerformanceDataPipeline(config.pipeline.sequence_length, target_cols)
    datasets = pipeline.run(
        csv_path,
        train_lower_percentile=config.pipeline.train_lower_percentile,
        train_upper_percentile=config.pipeline.train_upper_percentile,
    )

    # Derive feature list from the pipeline's schema, excluding non-numeric identifier columns.
    all_features = [
        col
        for col in PerformanceDataPipeline.REQUIRED_COLUMNS
        if col not in ("Timestamp", "Service")
    ] + PerformanceDataPipeline.DELTA_COLUMNS

    for service_name, data_split in datasets.items():
        logger.info(f"--- Service: {service_name} ---")

        X_train, y_train = data_split["train"]
        X_test, y_test = data_split["test"]

        logger.info(f"Train Shape: {X_train.shape} (Samples, Window, Features)")
        logger.info(f"Test Shape:  {X_test.shape}")

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

        output_size = y_train.shape[1]
        flat_input_size = X_train.shape[1] * X_train.shape[2]

        logger.info(f"Training PerformanceModel for {service_name}...")
        model = PerformanceModel(
            input_size=flat_input_size,
            output_size=output_size,
            hidden_size=config.model.hidden_size,
        )
        train_model(
            config,
            service_name,
            model,
            train_loader,
            test_loader,
            epochs=config.training.epochs,
            learning_rate=config.training.learning_rate,
        )

        logger.info(f"Loading the best saved model weights for {service_name}...")
        model_path = Path("models") / f"{service_name}.pth"
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
