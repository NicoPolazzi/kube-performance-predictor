import logging
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import DenseGATConv

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class HyperParameters:
    """Hyperparameters for the GNN-LSTM model training."""

    window_size: int = 10
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.0001
    hidden_channels: int = 16
    lstm_hidden_size: int = 64
    lstm_layers: int = 1
    dropout: float = 0.0
    val_split: float = 0.15
    test_split: float = 0.15


class MicroserviceDataset(Dataset):
    """PyTorch Dataset for sliding window sequences."""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DataPreprocessor:
    """
    Handles data loading, cleaning, log-transformation, and scaling.
    Maintains the state of the Scaler for inverse transformation.
    """

    def __init__(self, node_order: List[str], feature_cols: List[str]):
        self.node_order = node_order
        self.feature_cols = feature_cols
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.num_nodes = len(node_order)
        self.num_features = len(feature_cols)
        self.is_fitted = False

    def load_and_process(
        self, csv_path: str, split_ratios: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Loads CSV, enforces node order, fills gaps, and splits data.
        Returns: (train_tensor, val_tensor, test_tensor)
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        # Filter relevant nodes and enforce order
        df = df[df["Service"].isin(self.node_order)]
        df["Service"] = pd.Categorical(df["Service"], categories=self.node_order, ordered=True)

        # Create full grid (Timestamp x Service) to handle missing metrics
        timestamps = sorted(df["Timestamp"].unique())
        full_idx = pd.MultiIndex.from_product(
            [timestamps, self.node_order], names=["Timestamp", "Service"]
        )

        df = df.set_index(["Timestamp", "Service"]).reindex(full_idx)
        df = df[self.feature_cols].fillna(0)

        # Reshape to (Time, Nodes, Features)
        raw_data = df.values.reshape(len(timestamps), self.num_nodes, self.num_features)

        # Calculate split indices
        total_len = len(raw_data)
        val_size = int(split_ratios[0] * total_len)
        test_size = int(split_ratios[1] * total_len)
        train_size = total_len - val_size - test_size

        train_data = raw_data[:train_size]
        val_data = raw_data[train_size : train_size + val_size]
        test_data = raw_data[train_size + val_size :]

        return (
            self._transform(train_data, fit=True),
            self._transform(val_data, fit=False),
            self._transform(test_data, fit=False),
        )

    def _transform(self, data: np.ndarray, fit: bool = False) -> torch.Tensor:
        """Log transform -> Flatten -> Scale -> Reshape -> Tensor"""
        # 1. Log Transform (Robustness to outliers)
        data_log = np.log1p(data)

        # 2. Flatten for Scikit-Learn Scaler
        flat = data_log.reshape(-1, self.num_features)

        # 3. Scale
        if fit:
            flat_scaled = self.scaler.fit_transform(flat)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler must be fitted on training data first.")
            flat_scaled = self.scaler.transform(flat)

        # 4. Reshape and convert to Tensor
        return torch.tensor(flat_scaled.reshape(data.shape), dtype=torch.float32)

    def create_windows(self, data_tensor: torch.Tensor, window_size: int) -> MicroserviceDataset:
        """Generates sliding window sequences."""
        X_list, y_list = [], []

        # Ensure we don't go out of bounds
        if len(data_tensor) <= window_size:
            raise ValueError("Data length is smaller than window size.")

        for i in range(len(data_tensor) - window_size):
            X_list.append(data_tensor[i : i + window_size])
            y_list.append(data_tensor[i + window_size])

        X_tensor = torch.stack(X_list)
        y_tensor = torch.stack(y_list)
        return MicroserviceDataset(X_tensor, y_tensor)

    def inverse_transform_prediction(self, y_pred_scaled: np.ndarray) -> np.ndarray:
        """Reverses scaling and log transformation to get real units."""
        batch_size = y_pred_scaled.shape[0]

        # Ensure non-negative before inverse log
        y_pred_scaled = np.maximum(0, y_pred_scaled)

        flat_scaled = y_pred_scaled.reshape(-1, self.num_features)
        flat_log = self.scaler.inverse_transform(flat_scaled)
        flat_real = np.expm1(flat_log)

        return flat_real.reshape(batch_size, self.num_nodes, self.num_features)


# --- Model Definition ---


class DenseGATLSTM(nn.Module):
    """
    Hybrid model: Dense Graph Attention Network -> LSTM.
    Predicts the next system state based on a sequence of graph states.
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        adj_matrix: List[List[int]],
        params: HyperParameters,
    ):
        super(DenseGATLSTM, self).__init__()
        self.num_nodes = num_nodes
        self.params = params

        # Convert adjacency matrix to tensor
        self.register_buffer("adj", torch.tensor(adj_matrix, dtype=torch.float32))

        # 1. Spatial extraction (GAT)
        # Heads=4 is hardcoded in original notebook, making it configurable here is better
        self.gat = DenseGATConv(in_channels, params.hidden_channels, heads=4, concat=True)

        # 2. Temporal evolution (LSTM)
        # Input to LSTM is flattened (Nodes * (Hidden * Heads))
        gat_output_dim = num_nodes * (params.hidden_channels * 4)

        self.lstm = nn.LSTM(
            input_size=gat_output_dim,
            hidden_size=params.lstm_hidden_size,
            num_layers=params.lstm_layers,
            batch_first=True,
            dropout=params.dropout if params.lstm_layers > 1 else 0.0,
        )

        # 3. Prediction Head
        self.fc = nn.Linear(params.lstm_hidden_size, num_nodes * out_channels)
        self.adj: torch.Tensor

    def forward(self, x_seq: torch.Tensor):
        # x_seq: (Batch, Window, Nodes, Features)
        batch_size, window_size, _, _ = x_seq.size()

        # Repeat Adjacency for the batch
        adj_batch = self.adj.unsqueeze(0).repeat(batch_size, 1, 1)

        spatial_embeddings = []
        for t in range(window_size):
            xt = x_seq[:, t, :, :]  # (Batch, Nodes, Features)

            # GAT Forward
            gat_out = self.gat(xt, adj_batch)  # (Batch, Nodes, Hidden*Heads)

            # Flatten all nodes into one vector per timestep
            spatial_embeddings.append(gat_out.view(batch_size, -1))

        # Stack into sequence for LSTM: (Batch, Window, Flattened_Features)
        lstm_input = torch.stack(spatial_embeddings, dim=1)

        lstm_out, _ = self.lstm(lstm_input)

        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]

        # Decode
        out = self.fc(last_hidden)

        # Reshape back to node-level features
        return out.view(batch_size, self.num_nodes, -1)


class GNNTrainer:
    def __init__(self, model: nn.Module, params: HyperParameters):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
        self.criterion = nn.L1Loss()  # MAE Loss

        logger.info(f"Initialized Trainer on device: {self.device}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")

        logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(self.params.epochs):
            # Training Loop
            self.model.train()
            train_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            # Validation Loop
            avg_val_loss = self.evaluate(val_loader)
            history["val_loss"].append(avg_val_loss)

            # Logging & Checkpointing
            msg = ""
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                msg = "-> Saved Best Model"

            if (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.params.epochs} | "
                    f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} {msg}"
                )

        total_time = time.time() - start_time
        logger.info(f"Training finished in {total_time:.2f}s")
        return history

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(loader)
