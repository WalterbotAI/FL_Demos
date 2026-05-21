"""Shared model, dataset, and training logic for MedMNIST 3D."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MEDMNIST_ROOT", str(Path(".data/medmnist").resolve()))
warnings.filterwarnings("ignore", message="Failed to setup default root.")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import medmnist
from medmnist import INFO
from medmnist.evaluator import getACC

# ── Dataset metadata (read once at import) ────────────────────────────────────
DATA_FLAG  = "synapsemnist3d"
_info      = INFO[DATA_FLAG]
TASK       = _info["task"]
N_CHANNELS = _info["n_channels"]
N_CLASSES  = len(_info["label"])
DataClass  = getattr(medmnist, _info["python_class"])
DATA_ROOT = Path(os.environ["MEDMNIST_ROOT"]).resolve()


# ── Transform ─────────────────────────────────────────────────────────────────
class Transform3D:
    """Optional voxel intensity scaling."""

    def __init__(self, mul=None):
        self.mul = mul

    def __call__(self, voxel: np.ndarray) -> np.ndarray:
        if self.mul == "random":
            voxel = voxel * np.random.uniform(0, 1)
        elif self.mul is not None:
            voxel = voxel * float(self.mul)
        return voxel.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────
class MedMNIST3DDataset(Dataset):
    """Wraps a MedMNIST 3D split, normalising voxels to [0, 1].

    MedMNIST 3D returns images already in (C, D, H, W) format.
    """

    def __init__(self, medmnist_ds, transform=None):
        self.dataset   = medmnist_ds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.dataset[index]
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(img)
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(label.astype(int)),   # shape (1,)
        )


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(
    partition_id: int,
    num_partitions: int,
    train_bs: int = 64,
    val_bs: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    """Return train/val DataLoaders for the given client partition."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    raw_train = DataClass(split="train", download=True, root=str(DATA_ROOT))
    raw_val   = DataClass(split="val",   download=True, root=str(DATA_ROOT))

    n_total   = len(raw_train)
    per_slice = n_total // num_partitions
    start     = partition_id * per_slice
    end       = start + per_slice if partition_id < num_partitions - 1 else n_total

    train_ds = MedMNIST3DDataset(raw_train, transform=Transform3D())
    val_ds   = MedMNIST3DDataset(raw_val,   transform=Transform3D())

    trainloader = DataLoader(
        Subset(train_ds, list(range(start, end))),
        batch_size=train_bs, shuffle=True, num_workers=0,
    )
    valloader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=0)

    return trainloader, valloader


# ── Model ─────────────────────────────────────────────────────────────────────
class Net3D(nn.Module):
    """3D CNN for volumetric MedMNIST classification (input: 28×28×28)."""

    def __init__(self, in_channels: int = N_CHANNELS, num_classes: int = N_CLASSES):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(in_channels, 16, 3), nn.BatchNorm3d(16), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv3d(16, 16, 3), nn.BatchNorm3d(16), nn.ReLU(), nn.MaxPool3d(2, 2))
        self.layer3 = nn.Sequential(nn.Conv3d(16, 64, 3), nn.BatchNorm3d(64), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv3d(64, 64, 3), nn.BatchNorm3d(64), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv3d(64, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(), nn.MaxPool3d(2, 2))
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 128),             nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ── Parameter helpers ─────────────────────────────────────────────────────────
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = {
        k: torch.tensor(v)
        for k, v in zip(model.state_dict().keys(), parameters)
    }
    model.load_state_dict(state_dict, strict=True)


# ── Train ─────────────────────────────────────────────────────────────────────
def train(
    model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    local_epochs: int = 1,
    lr: float = 0.001,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    model.to(device)

    losses = []
    for _ in range(local_epochs):
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if TASK == "multi-label, binary-class":
                loss = criterion(outputs, targets.squeeze().float())
            else:
                loss = criterion(outputs, targets.squeeze().long())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return {"train_loss": float(np.mean(losses))}


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(
    model: nn.Module,
    valloader: DataLoader,
    device: torch.device,
) -> Tuple[float, int, dict]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)

    losses, y_scores, y_labels = [], [], []
    with torch.no_grad():
        for inputs, targets in valloader:
            outputs = model(inputs.to(device))
            loss    = criterion(outputs, targets.to(device).squeeze().long())
            losses.append(loss.item())
            y_scores.append(F.softmax(outputs, dim=1).cpu().numpy())
            y_labels.append(targets.numpy())

    y_score = np.concatenate(y_scores, axis=0)
    y_true  = np.concatenate(y_labels, axis=0)
    acc     = getACC(y_true, y_score, TASK)

    return float(np.mean(losses)), len(valloader.dataset), {"accuracy": float(acc)}
