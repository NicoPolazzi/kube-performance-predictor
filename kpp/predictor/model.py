from typing import cast

import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, gru_out: torch.Tensor) -> torch.Tensor:
        # gru_out: (batch, seq_len, hidden_size)
        weights = torch.softmax(self.attn(gru_out), dim=1)  # (batch, seq_len, 1)
        return cast(torch.Tensor, (weights * gru_out).sum(dim=1))  # (batch, hidden_size)


class PerformancesGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        use_attention: bool = True,
    ):
        """
        Args:
            input_size: Number of features in X (e.g., 4: User Count, CPU, Resp Time, Throughput).
            hidden_size: Number of neurons in the hidden state.
            output_size: Number of target features to predict in y.
            num_layers: Number of stacked GRU layers.
            dropout: Dropout probability (only applies if num_layers > 1).
            use_attention: If True, use learned weighted sum over all hidden states (AMDLN).
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        if use_attention:
            self.attention = Attention(hidden_size)
        self.hidden_fc = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, sequence_length, input_size)
        """
        out, _ = self.gru(x)  # output features from the last layer of the GRU for each time step.
        if self.use_attention:
            context = self.attention(out)
        else:
            context = out[:, -1, :]  # last timestep only
        context = torch.relu(self.hidden_fc(context))
        return cast(torch.Tensor, self.fc(context))


class LinearBaseline(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) — use only last timestep
        return cast(torch.Tensor, self.fc(x[:, -1, :]))
