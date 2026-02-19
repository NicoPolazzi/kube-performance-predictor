from typing import cast

import torch
import torch.nn as nn


class PerformancesGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Args:
            input_size: Number of features in X (e.g., 4: User Count, CPU, Resp Time, Throughput).
            hidden_size: Number of neurons in the hidden state.
            output_size: Number of target features to predict in y.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability (only applies if num_layers > 1).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, sequence_length, input_size)
        """
        out, _ = self.gru(x)  # output features from the last layer of the GRU for each time step.
        last_time_step = out[
            :, -1, :
        ]  # We only care about the prediction at the very end of the sequence window.
        return cast(torch.Tensor, self.fc(last_time_step))


class KubernetesPredictorMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, sequence_length: int):
        super().__init__()

        # Calculate the total number of features after flattening the 2D window
        self.flattened_size = input_size * sequence_length
        self.hidden_size = hidden_size

        # The fully connected (dense) layers
        self.fc1 = nn.Linear(self.flattened_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Final output layer mapping to your 3 targets
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, sequence_length, input_features)
        """
        # Flatten the 3D tensor into a 2D tensor
        # E.g., from (32, 5, 4) -> (32, 20)
        x = x.view(x.size(0), -1)

        # Pass through the network
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.fc3(x)


class KubernetesPredictorCNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Args:
            input_size: Number of features in X (e.g., 4: User Count, CPU, etc.).
            hidden_size: Number of filters/channels in the convolution layers.
            output_size: Number of target features to predict in y (e.g., 3).
        """
        super().__init__()

        # 1. First Convolutional Layer
        # kernel_size=3 means it looks at 3 minutes of data at a time.
        # padding=1 ensures the output sequence length stays the same as the input.
        self.conv1 = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU()

        # 2. Second Convolutional Layer (Finds more complex compound shapes)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1
        )

        # 3. Global Average Pooling
        # This is a magic layer! It squashes the entire time sequence down to 1 value
        # per channel. This means your model will not crash even if you change
        # sequence_length from 5 to 10 in your data pipeline later.
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 4. Final Output Mapping
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, sequence_length, input_features)
        """
        # CRITICAL PYTORCH TRICK:
        # Linear/GRU layers expect: (Batch, Sequence, Features)
        # Conv1d layers expect:     (Batch, Features, Sequence)
        # We must swap the last two dimensions using permute.
        x = x.permute(0, 2, 1)

        # Extract local patterns (the "magnifying glass")
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        # Compress the time dimension down to a single point
        x = self.global_pool(x)  # Shape becomes: (Batch, hidden_size, 1)

        # Remove the empty last dimension to make it a flat vector
        x = x.squeeze(-1)  # Shape becomes: (Batch, hidden_size)

        # Make the final prediction
        return self.fc(x)
