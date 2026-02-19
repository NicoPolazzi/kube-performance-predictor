import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from kpp.predictor.model import PerformancesGRU
from kpp.predictor.pipeline import PerformancesDataPipeline
from kpp.predictor.visualizer import evaluate_and_plot


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
):
    """Handles the training and validation loops."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate
    )  # Here we give the references to the model parameters

    """
    FIXME: Probably we need a way to store the best model with its hyper pamaters
    """
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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}"
            )


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

        model = PerformancesGRU(
            input_size=input_size,
            hidden_size=64,
            output_size=output_size,
            num_layers=1,
            dropout=0.2,
        )

        train_model(model, train_loader, test_loader, epochs=50)

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
