"""PyTorch tabular model: entity embeddings for categoricals + an MLP.

Why embeddings? One-hot encoding 160+ countries makes a huge sparse input.
An embedding learns a small dense vector per category instead, so similar
countries/regions can end up close together.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from shipsense.config import CATEGORICAL, NUMERIC, SEED


@dataclass
class TabularEncoder:
    """Maps categories to integer ids and standardises numerics (fit on train only)."""

    vocab: dict = field(default_factory=dict)
    mean: pd.Series | None = None
    std: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "TabularEncoder":
        # id 0 is reserved for "unknown / unseen in training"
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].astype(str).unique()))}
                      for c in CATEGORICAL}
        self.mean = df[NUMERIC].mean()
        self.std = df[NUMERIC].std().replace(0, 1)
        return self

    def transform(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        cats = np.stack([df[c].astype(str).map(self.vocab[c]).fillna(0).astype(int).to_numpy()
                         for c in CATEGORICAL], axis=1)
        nums = ((df[NUMERIC] - self.mean) / self.std).to_numpy(dtype=np.float32)
        return torch.tensor(cats, dtype=torch.long), torch.tensor(nums)

    def cardinalities(self) -> list[int]:
        return [len(self.vocab[c]) + 1 for c in CATEGORICAL]


class LateDeliveryNet(nn.Module):
    def __init__(self, cardinalities: list[int], n_numeric: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        # rule of thumb: embedding size ~ min(16, (cardinality + 1) // 2)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, min(16, (n + 1) // 2)) for n in cardinalities])
        emb_dim = sum(e.embedding_dim for e in self.embeddings)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + n_numeric, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.BatchNorm1d(hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        embs = [emb(cats[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [nums], dim=1)
        return self.mlp(x).squeeze(1)  # raw logits; sigmoid is applied in the loss


class TorchClassifier:
    """sklearn-style wrapper (fit / predict_proba) with early stopping on validation loss."""

    def __init__(self, epochs: int = 30, batch_size: int = 512, lr: float = 1e-3,
                 weight_decay: float = 1e-5, patience: int = 4):
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.weight_decay, self.patience = weight_decay, patience
        self.history: list[dict] = []

    def fit(self, X_train: pd.DataFrame, y_train, X_val: pd.DataFrame, y_val):
        torch.manual_seed(SEED)
        self.encoder = TabularEncoder().fit(X_train)
        tr = TensorDataset(*self.encoder.transform(X_train), torch.tensor(np.asarray(y_train), dtype=torch.float32))
        va_c, va_n = self.encoder.transform(X_val)
        va_y = torch.tensor(np.asarray(y_val), dtype=torch.float32)
        loader = DataLoader(tr, batch_size=self.batch_size, shuffle=True)

        self.model = LateDeliveryNet(self.encoder.cardinalities(), len(NUMERIC))
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total = 0.0
            for cats, nums, y in loader:
                opt.zero_grad()
                loss = loss_fn(self.model(cats, nums), y)
                loss.backward()
                opt.step()
                total += loss.item() * len(y)
            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(va_c, va_n), va_y).item()
            self.history.append({"epoch": epoch, "train_loss": total / len(tr), "val_loss": val_loss})
            if val_loss < best - 1e-4:
                best, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:  # early stopping: val loss stopped improving
                    break
        self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            p = torch.sigmoid(self.model(*self.encoder.transform(X))).numpy()
        return np.stack([1 - p, p], axis=1)
