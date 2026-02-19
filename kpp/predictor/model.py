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
