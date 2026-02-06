import logging
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import StandardScaler
from torch.utils.data import Dataset, TensorDataset

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
    """
    PyTorch Dataset with on-the-fly Noise Injection.
    """

    def __init__(self, X: torch.Tensor, y: torch.Tensor, noise_level: float = 0.0):
        self.X = X
        self.y = y
        self.noise_level = noise_level

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        y_sample = self.y[idx]

        # Add noise ONLY if noise_level is set (Training Mode)
        if self.noise_level > 0:
            # Generate random noise (mean=0, std=noise_level)
            noise = torch.randn_like(x_sample) * self.noise_level
            x_sample = x_sample + noise

        return x_sample, y_sample


class DataPreprocessor:
    def __init__(self, node_order: List[str], feature_cols: List[str]):
        self.node_order = node_order  # List of service names (used for filtering/sorting)
        self.feature_cols = feature_cols
        self.scaler = StandardScaler()

    def load_and_process(
        self, csv_path: str, split_ratios: Tuple[float, float] = (0.0, 0.0)
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Loads CSV and splits based on USER COUNT INTERPOLATION.
        """
        logger.info(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)

        # 1. Clean Data
        df[self.feature_cols] = df[self.feature_cols].interpolate(method="linear").bfill().ffill()

        train_dfs = []
        # val_dfs = []  # We might not need Val if we use Test for validation
        test_dfs = []

        # 2. Identify Unique User Counts
        #    We assume the file contains discrete steps: 50, 100, 150, 200...
        unique_users = sorted(df["User Count"].unique())
        logger.info(f"Found {len(unique_users)} unique user levels: {unique_users}")

        # 3. Create the Split Mask (Strided)
        #    Pattern: Train, Train, Test ... (66% Train, 33% Test)
        #    Or your request: Odd (Train), Even (Test) -> (50% Train, 50% Test)

        train_users = []
        test_users = []

        for i, u_count in enumerate(unique_users):
            # Your Logic: "Odd numbers (indices) for training, Even for test"
            # To get MORE training data, we can say:
            # "If index % 3 == 0, it's Test. Else, it's Train."
            if (i + 1) % 3 == 0:
                test_users.append(u_count)
            else:
                train_users.append(u_count)

        logger.info(f"Training on User Counts: {train_users}")
        logger.info(f"Testing on User Counts: {test_users}")

        # 4. Filter and Fit Scaler
        #    We fit scaler ONLY on training user counts
        train_raw_df = df[df["User Count"].isin(train_users)]
        self.scaler.fit(train_raw_df[self.feature_cols].values)

        # 5. Build the Lists (One array per service, per load level)
        #    This is crucial: We treat "Service A at 100 Users" as one continuous block.

        for service in df["Service"].unique():
            svc_df = df[df["Service"] == service]

            # Add Training Blocks
            for u in train_users:
                subset = svc_df[svc_df["User Count"] == u]
                if not subset.empty:
                    # Sort by time to keep temporal consistency within the block
                    data = subset.sort_values("Timestamp")[self.feature_cols].values
                    train_dfs.append(self.scaler.transform(data))

            # Add Test Blocks
            for u in test_users:
                subset = svc_df[svc_df["User Count"] == u]
                if not subset.empty:
                    data = subset.sort_values("Timestamp")[self.feature_cols].values
                    test_dfs.append(self.scaler.transform(data))

        # Note: We return empty list for 'Val' since we are doing a 2-way split here.
        # You can split 'train_dfs' further if you strictly need a validation set.
        return train_dfs, [], test_dfs

    def create_dataset(
        self, df_list: List[np.ndarray], window_size: int, noise_level: float = 0.0
    ) -> TensorDataset:
        """
        Creates windows for EACH service separately, then concatenates them.
        Prevents windows from crossing service boundaries.
        """
        all_X = []
        all_y = []

        for data in df_list:
            # 1. Convert to Tensor
            data_tensor = torch.FloatTensor(data)

            # 2. Skip if data is shorter than window
            if len(data_tensor) <= window_size:
                continue

            X_windows = []
            y_targets = []
            for i in range(len(data_tensor) - window_size):
                X_windows.append(data_tensor[i : i + window_size])
                y_targets.append(data_tensor[i + window_size, 1:])  # Target is next step

            if not X_windows:
                continue

            # Stack and add noise
            chunk_X = torch.stack(X_windows)
            if noise_level > 0:
                chunk_X += torch.randn_like(chunk_X) * noise_level

            all_X.append(chunk_X)
            all_y.append(torch.stack(y_targets))

        # Concatenate all chunks
        final_X = torch.cat(all_X)
        final_y = torch.cat(all_y)

        return TensorDataset(final_X, final_y)


class MetricForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.2):
        """
        Args:
            input_size (int): Number of input features per time step (e.g., CPU, RT, Throughput, Users = 4).
            hidden_size (int): Number of neurons in the GRU hidden state.
            output_size (int): Number of metrics to predict (e.g., CPU, RT, Throughput = 3).
            num_layers (int): Number of GRU layers stacked.
        """
        super(MetricForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [Batch, Window_Size, Features]
        out, _ = self.gru(x)

        # last_time_step shape: (batch_size, hidden_size)
        last_time_step = out[:, -1, :]

        # Predict the next metrics
        prediction = self.fc(last_time_step)
        return prediction

    def predict(self, x_input):
        """
        A helper method for Inference/Plotting.
        Handles mode switching and data conversion automatically.
        """
        # 1. Ensure model is in evaluation mode (turns off Dropout)
        self.eval()

        # 2. Handle Input: If it's a Numpy array, convert to Tensor
        if hasattr(x_input, "values"):  # Checks if it's a DataFrame or Series
            x_tensor = torch.FloatTensor(x_input.values)
        elif isinstance(x_input, np.ndarray):
            x_tensor = torch.FloatTensor(x_input)
        else:
            x_tensor = x_input

        # 3. Check for Batch Dimension
        # If user passes a single window (30, 4), we need to make it (1, 30, 4)
        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)

        # 4. Run Inference without Gradients (Saves memory/speed)
        with torch.no_grad():
            prediction_tensor = self(x_tensor)  # Calls forward() internally

        # 5. Return a Numpy array (Standard for plotting libraries)
        return prediction_tensor.cpu().numpy()
