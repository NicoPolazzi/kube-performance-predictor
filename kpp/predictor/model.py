from typing import cast

import torch
import torch.nn as nn


class PerformanceModel(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size) — flatten full window
        return cast(torch.Tensor, self.net(x.reshape(x.size(0), -1)))
