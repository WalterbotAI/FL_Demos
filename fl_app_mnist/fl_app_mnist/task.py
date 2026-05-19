"""MNIST task: CNN model, data loading, train/evaluate utilities."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

DATA_DIR = Path.home() / ".flwr_demo" / "mnist"

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


class Net(nn.Module):
    """Small CNN for MNIST: 2 conv layers + 2 fully connected."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(torch.flatten(x, 1))
        x = F.relu(self.fc1(x))
        return self.fc2(self.dropout(x))


def load_data(
    partition_id: int,
    num_partitions: int = 2,
    max_samples: int = 500,
) -> tuple[DataLoader, DataLoader]:
    """Return train/test DataLoaders for one federation partition.

    MNIST is downloaded once to DATA_DIR and cached for subsequent runs.
    Each partition gets an equal, non-overlapping slice of the training set
    capped at ``max_samples`` to keep rounds fast on CPU.
    All partitions share the full test set (10 000 samples).
    """
    train_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test_full = datasets.MNIST(DATA_DIR, train=False, download=True, transform=TRANSFORM)

    n = len(train_full)
    per_partition = n // num_partitions
    start = partition_id * per_partition
    end = min(start + max_samples, start + per_partition)
    train_subset = Subset(train_full, list(range(start, end)))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False)
    return train_loader, test_loader


def get_parameters(model: Net) -> list:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: Net, parameters: list) -> None:
    state_dict = dict(zip(model.state_dict().keys(), map(torch.tensor, parameters)))
    model.load_state_dict(state_dict, strict=True)


def train(
    model: Net,
    train_loader: DataLoader,
    epochs: int = 1,
    device: torch.device | str = "cpu",
) -> tuple[list, int, dict]:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()
    return get_parameters(model), len(train_loader.dataset), {}


def evaluate(
    model: Net,
    test_loader: DataLoader,
    device: torch.device | str = "cpu",
) -> tuple[float, int, dict]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
    n = len(test_loader.dataset)
    return total_loss / len(test_loader), n, {"accuracy": correct / n}
