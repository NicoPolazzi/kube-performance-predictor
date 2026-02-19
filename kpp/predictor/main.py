import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kpp.predictor.model import KubernetesPredictorMLP
from kpp.predictor.pipeline import PerformancesDataPipeline
from kpp.predictor.visualizer import evaluate_and_plot


def train_model(
    service_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
):
    """Handles the training and validation loops."""
    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"gru_{service_name}.pth"
    config_path = out_dir / f"config_{service_name}.json"

    best_test_loss = float("inf")

    # if config_path.exists():
    #     with open(config_path, "r") as f:
    #         try:
    #             old_config = json.load(f)
    #             best_test_loss = old_config.get("best_test_loss", float("inf"))
    #         except json.JSONDecodeError:
    #             print("Warning: config file is corrupted. Starting fresh.")

    # if config_path.exists():
    #     with open(config_path, "r") as f:
    #         try:
    #             old_config = json.load(f)
    #             is_same_arch = (
    #                 old_config.get("num_layers") == model.num_layers
    #                 and old_config.get("hidden_size") == model.hidden_size
    #             )

    #             if is_same_arch:
    #                 best_test_loss = old_config.get("best_test_loss", float("inf"))
    #                 print(
    #                     f"Found existing model for {service_name}. Historical best to beat: {best_test_loss:.6f}"
    #                 )
    #             else:
    #                 print(
    #                     f"Architecture change detected for {service_name}. Ignoring old checkpoints."
    #                 )

    #         except json.JSONDecodeError:
    #             print("Warning: config file is corrupted. Starting fresh.")

    hyperparams = {
        "service": service_name,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "hidden_size": getattr(model, "hidden_size", "N/A"),
        "num_layers": getattr(model, "num_layers", "N/A"),
    }
    with open(config_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    print(f"Saved hyperparameters to {config_path}")

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

        train_loss /= train_total_samples
        scheduler.step(test_loss)

        model.eval()
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                test_loss += loss.item() * batch_X.size(0)
                test_total_samples += batch_X.size(0)

        test_loss /= test_total_samples

        """
        FIXME: Probably we want to compute the RMSE or just don't compute it at all
        """
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
            print(
                f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
            )

    print(f"Training complete. Best model weights are maintained at: {model_path}")


def main():
    csv_path = "performance_results_medium.csv"
    sequence_length = 5
    target_cols = [
        "Response Time (s)",
        "Throughput (req/s)",
        "CPU Usage",
    ]

    pipeline = PerformancesDataPipeline(sequence_length, target_cols)
    datasets = pipeline.run(csv_path, split_ratio=0.8)

    for service_name, data_split in datasets.items():
        print(f"\n--- Service: {service_name} ---")

        X_train, y_train = data_split["train"]
        X_test, y_test = data_split["test"]

        print(f"Train Shape: {X_train.shape} (Samples, Window, Features)")
        print(f"Test Shape:  {X_test.shape}")

        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        input_size = X_train.shape[2]
        output_size = y_train.shape[1]

        # model = PerformancesGRU(
        #     input_size=input_size,
        #     hidden_size=32,
        #     output_size=output_size,
        #     num_layers=2,
        #     dropout=0.1,
        # )

        # model = KubernetesPredictorCNN(
        #     input_size=input_size,
        #     hidden_size=512,  # Start with 64 filters
        #     output_size=output_size,
        # )

        model = KubernetesPredictorMLP(
            input_size=input_size,
            hidden_size=128,
            output_size=output_size,
            sequence_length=sequence_length,
        )

        train_model(service_name, model, train_loader, test_loader, epochs=500)

        print(f"Loading the best saved weights for {service_name}...")
        model_path = Path("models") / f"gru_{service_name}.pth"
        model.load_state_dict(torch.load(model_path, weights_only=True))

        print(f"Evaluating and plotting {service_name}...")

        all_features = ["User Count", "Response Time (s)", "Throughput (req/s)", "CPU Usage"]
        service_scaler = pipeline.scalers[service_name]

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
