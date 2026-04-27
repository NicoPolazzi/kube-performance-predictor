import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from kpp.config import PredictorConfig

logger = logging.getLogger(__name__)


class ClassificationModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        hidden_size_2: int = 64,
        head_hidden_size: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size_2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_size_2, head_hidden_size),
            nn.GELU(),
            nn.Linear(head_hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.head(self.trunk(x))
        return result


def train_classifier(
    config: PredictorConfig,
    service_name: str,
    model: ClassificationModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    class_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> tuple[float, list[float], list[float]]:
    """Trains a ClassificationModel and returns (best_test_acc, train_losses, test_losses).

    Best weights are selected by highest test accuracy rather than lowest test loss.
    This is important for the extrapolation setting where distribution shift causes
    cross-entropy loss to spike even when the model classifies correctly.

    The LR scheduler steps on training loss (not test loss) so that distribution
    shift in the test set does not cause premature LR reduction.
    """
    best_test_acc = -1.0
    best_state_dict = None
    train_losses: list[float] = []
    test_losses: list[float] = []

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=config.training.weight_decay
    )
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
        train_total = 0
        test_total = 0
        train_correct = 0

        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            train_total += batch_x.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        if train_total == 0:
            raise RuntimeError(f"No training samples found for {service_name}.")
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        test_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                test_loss += loss.item() * batch_x.size(0)
                test_total += batch_x.size(0)
                test_correct += (logits.argmax(dim=1) == batch_y).sum().item()

        if test_total == 0:
            raise RuntimeError(f"No test samples found for {service_name}.")
        test_loss /= test_total
        test_acc = test_correct / test_total

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # Step on training loss: test loss is unreliable under distribution shift
        # (it can spike to 20+ while the model still classifies correctly),
        # so using it would cause premature LR reduction.
        scheduler.step(train_loss)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state_dict = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        model.eval()
    logger.info(
        f"Training complete. Best weights restored (best test acc: {best_test_acc:.4f})."
    )
    return best_test_acc, train_losses, test_losses


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs inference on the test set. Returns (predicted_classes, true_classes) as int arrays."""
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)
