"""Baseline LSTM model scaffold for YouBike demand forecasting."""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMConfig:
    input_size: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    horizon: int = 12  # 12 * 5min = 60 min horizon by default
    history: int = 24
    batch_size: int = 32
    epochs: int = 5
    learning_rate: float = 1e-3
    device: str = "cpu"


class BaselineLSTM(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout,
        )
        self.head = nn.Linear(config.hidden_size, config.horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.head(last_hidden)


def generate_synthetic_dataset(
    num_samples: int,
    history: int,
    horizon: int,
    input_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a synthetic dataset mimicking bike demand oscillations."""
    time_steps = torch.linspace(0, 2 * torch.pi, history + horizon)
    inputs = torch.zeros(num_samples, history, input_size)
    targets = torch.zeros(num_samples, horizon)

    for i in range(num_samples):
        phase = torch.rand(1) * 2 * torch.pi
        freq = torch.randint(1, 4, (1,), dtype=torch.float32)
        base_signal = 20 + 5 * torch.sin(freq * time_steps + phase)
        noise = torch.randn(history + horizon) * 0.5
        demand = base_signal + noise

        feature_stack = []
        for feature_idx in range(input_size):
            drift = torch.linspace(0, 0.2 * (feature_idx + 1), history + horizon)
            feature_noise = torch.randn(history + horizon) * (0.1 + feature_idx * 0.02)
            feature_stack.append(demand + drift + feature_noise)

        features = torch.stack(feature_stack, dim=-1)
        inputs[i] = features[:history]
        targets[i] = demand[history:history + horizon]

    return inputs, targets


def train_baseline(config: LSTMConfig) -> Dict[str, float]:
    """Train the baseline model on synthetic data and report loss stats."""
    device = torch.device(config.device)
    model = BaselineLSTM(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    inputs, targets = generate_synthetic_dataset(
        num_samples=512,
        history=config.history,
        horizon=config.horizon,
        input_size=config.input_size,
    )

    split = int(0.8 * inputs.size(0))
    train_x, val_x = inputs[:split], inputs[split:]
    train_y, val_y = targets[:split], targets[split:]

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
    )

    loss_history = []
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)

        epoch_loss /= train_x.size(0)
        loss_history.append(epoch_loss)
        logging.info("Epoch %d | loss=%.4f", epoch + 1, epoch_loss)

    with torch.no_grad():
        val_pred = model(val_x.to(device)).cpu()
    val_mse = criterion(val_pred, val_y).item()
    val_rmse = torch.sqrt(torch.tensor(val_mse)).item()

    return {
        "train_loss_final": loss_history[-1],
        "train_loss_initial": loss_history[0],
        "val_rmse": val_rmse,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = LSTMConfig()
    metrics = train_baseline(config)
    logging.info("Baseline training complete | metrics=%s", asdict(config))
    logging.info(
        "Training summary | initial_loss=%.4f final_loss=%.4f val_rmse=%.4f",
        metrics["train_loss_initial"],
        metrics["train_loss_final"],
        metrics["val_rmse"],
    )


if __name__ == "__main__":
    main()
