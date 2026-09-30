import random
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def save_checkpoint(path: str, model, optimizer=None, extra: dict | None = None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, model, optimizer=None, map_location=None):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "min"):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        if self.best is None:
            self.best = value
            return False
        improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
