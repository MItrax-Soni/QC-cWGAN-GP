"""
Early Stopping Engine.
Monitors validation metric (loss or macro F1) and halts training when patience is exceeded.
"""

from typing import Optional
import numpy as np
import torch


class EarlyStopping:
    """
    Early stopping handler that monitors validation metrics and tracks model checkpointing.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, val_metric: float) -> bool:
        score = val_metric if self.mode == "max" else -val_metric

        if self.best_score is None:
            self.best_score = score
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False
