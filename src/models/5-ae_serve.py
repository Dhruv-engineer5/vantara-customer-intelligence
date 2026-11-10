"""Torch-free autoencoder serving for the deployed demo.

The autoencoder is trained with torch (src/models/deep.py). For the deployment
image its weights are converted to models_artifacts/autoencoder_weights.json and
inference runs in NumPy, so the serving container does not need torch (~176 MB
RSS). Validated against the torch model: anomaly flags match on all 5,248
scored customers; max reconstruction-error diff 1.3e-07. See README "Deployment".
"""
import json
from pathlib import Path

import numpy as np


class NumpyAutoEncoder:
    """NumPy forward pass matching src.models.deep.AutoEncoder (Linear-ReLU-Linear / Linear-ReLU-Linear)."""

    def __init__(self, json_path: str | Path) -> None:
        w = json.loads(Path(json_path).read_text())
        self.p = {k: np.array(v, dtype=np.float64) for k, v in w.items()}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        e = np.maximum(x @ p["enc.0.weight"].T + p["enc.0.bias"], 0.0)
        e = e @ p["enc.2.weight"].T + p["enc.2.bias"]
        d = np.maximum(e @ p["dec.0.weight"].T + p["dec.0.bias"], 0.0)
        return d @ p["dec.2.weight"].T + p["dec.2.bias"]


def recon_error(model: NumpyAutoEncoder, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error (same formula as the torch version)."""
    x = np.asarray(X, dtype=np.float64)
    return ((model(x) - x) ** 2).mean(axis=1)
