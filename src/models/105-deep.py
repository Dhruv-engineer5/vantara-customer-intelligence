"""PyTorch models: feed-forward ANN, LSTM on purchase sequences, autoencoder for anomalies."""

from __future__ import annotations

from typing import Any

import copy

import numpy as np
import torch
from torch import nn

from src.models.common import Timer


def set_seed(seed: int) -> None:
    """Fix torch/numpy seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)


class ANN(nn.Module):
    """Feed-forward churn classifier with batch norm and dropout."""

    def __init__(self, n_in: int, hidden: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on a batch."""
        return self.net(x).squeeze(-1)


class LSTMClassifier(nn.Module):
    """LSTM over the last purchases of each customer, predicting churn probability."""

    def __init__(self, n_feat: int, hidden: int, dropout: float, n_out: int = 1) -> None:
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, n_out)
        self.n_out = n_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on a batch."""
        _, (h, _) = self.lstm(x)
        o = self.out(self.drop(h[-1]))
        return o.squeeze(-1) if self.n_out == 1 else o


class AutoEncoder(nn.Module):
    """Small autoencoder; reconstruction error above a validated threshold flags anomalies."""

    def __init__(self, n_in: int, latent: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n_in, 16), nn.ReLU(), nn.Linear(16, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 16), nn.ReLU(), nn.Linear(16, n_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network on a batch."""
        return self.dec(self.enc(x))


def fit_binary(
    model: nn.Module,
    Xtr: Any,
    ytr: Any,
    Xva: Any,
    yva: Any,
    cfg: dict,
    pos_weight: float = 1.0,
    task: str = "binary",
) -> tuple[nn.Module, list, list]:
    """Train with early stopping on validation loss. task: binary (BCE), reg (MSE) or multi (cross-entropy).

    Returns (best model, history dict, seconds).
    """
    d = cfg["models"]["deep"]
    l1 = float(d.get("l1", 0.0))  # L1 penalty added to the training loss on weight matrices
    opt = torch.optim.Adam(model.parameters(), lr=d["lr"], weight_decay=float(d.get("l2", 0.0)))  # L2 penalty
    if task == "binary":
        loss_fn, ydt = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight)), torch.float32
    elif task == "reg":
        loss_fn, ydt = nn.MSELoss(), torch.float32
    else:
        loss_fn, ydt = nn.CrossEntropyLoss(), torch.long
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=ydt)
    Xva_t, yva_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(yva, dtype=ydt)
    best, best_state, bad = np.inf, None, 0
    hist = {"train_loss": [], "val_loss": []}
    with Timer() as t:
        for _ in range(d["epochs"]):
            model.train()
            perm = torch.randperm(len(Xtr_t))
            tot = 0.0
            for i in range(0, len(perm), d["batch_size"]):
                b = perm[i : i + d["batch_size"]]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                data_loss = loss_fn(model(Xtr_t[b]), ytr_t[b])
                loss = data_loss + l1 * sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
                loss.backward()
                opt.step()
                tot += float(data_loss.detach()) * len(b)
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(model(Xva_t), yva_t))
            hist["train_loss"].append(tot / len(perm))
            hist["val_loss"].append(vl)
            if vl < best - 1e-5:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= d["patience"]:
                    break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist, t.seconds


def predict_proba(model: nn.Module, X: Any) -> np.ndarray:
    """Sigmoid probabilities from a trained binary model."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def predict_raw(model: nn.Module, X: Any) -> np.ndarray:
    """Raw network output (regression value or class logits)."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def fit_autoencoder(Xtr: Any, Xva: Any, cfg: dict) -> tuple[nn.Module, list, list]:
    """Train the autoencoder on scaled features (unsupervised); returns (model, threshold, history, seconds)."""
    d = cfg["models"]["deep"]
    model = AutoEncoder(Xtr.shape[1], d["ae_latent"])
    opt = torch.optim.Adam(model.parameters(), lr=d["lr"])
    mse = nn.MSELoss()
    tr, va = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xva, dtype=torch.float32)
    best, best_state, bad, hist = np.inf, None, 0, {"train_loss": [], "val_loss": []}
    with Timer() as t:
        for _ in range(d["epochs"]):
            model.train()
            perm = torch.randperm(len(tr))
            tot = 0.0
            for i in range(0, len(perm), d["batch_size"]):
                b = perm[i : i + d["batch_size"]]
                opt.zero_grad()
                loss = mse(model(tr[b]), tr[b])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            model.eval()
            with torch.no_grad():
                vl = float(mse(model(va), va))
            hist["train_loss"].append(tot / len(perm))
            hist["val_loss"].append(vl)
            if vl < best - 1e-6:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= d["patience"]:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        err = ((model(va) - va) ** 2).mean(dim=1).numpy()
    thr = float(np.percentile(err, d["anomaly_percentile"]))
    return model, thr, hist, t.seconds


def recon_error(model: nn.Module, X: Any) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    model.eval()
    x = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        return ((model(x) - x) ** 2).mean(dim=1).numpy()
