"""
Experiment Logging and Metrics Tracking Engine.
Maintains persistent CSV logs, training metrics records, and formatted terminal outputs.
"""

import os
import csv
import json
import time
from typing import Dict, Any, List


class ExperimentLogger:
    """
    Handles CSV metric logging, experiment configuration archival, and training progress.
    """
    def __init__(self, experiment_dir: str, config: Dict[str, Any]):
        self.experiment_dir = experiment_dir
        self.config = config
        os.makedirs(experiment_dir, exist_ok=True)

        self.csv_path = os.path.join(experiment_dir, "metrics.csv")
        self.config_path = os.path.join(experiment_dir, "config.json")
        self._init_csv()
        self._save_config()

    def _init_csv(self):
        """Initializes the CSV file with fieldnames if it doesn't already exist."""
        if not os.path.exists(self.csv_path):
            fieldnames = [
                "epoch",
                "global_step",
                "generator_loss",
                "critic_loss",
                "wasserstein_estimate",
                "gradient_penalty",
                "real_score",
                "fake_score",
                "epoch_duration_sec",
                "ram_used_percent",
                "ram_available_gb"
            ]
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def _save_config(self):
        """Saves experiment configuration JSON."""
        with open(self.config_path, "w") as f:
            json.dump(self.config, f, indent=2)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Appends a completed epoch's metrics to CSV."""
        row = {
            "epoch": epoch,
            "global_step": metrics.get("global_step", 0),
            "generator_loss": round(metrics.get("generator_loss", 0.0), 4),
            "critic_loss": round(metrics.get("critic_loss", 0.0), 4),
            "wasserstein_estimate": round(metrics.get("wasserstein_estimate", 0.0), 4),
            "gradient_penalty": round(metrics.get("gradient_penalty", 0.0), 4),
            "real_score": round(metrics.get("real_score", 0.0), 4),
            "fake_score": round(metrics.get("fake_score", 0.0), 4),
            "epoch_duration_sec": round(metrics.get("epoch_duration_sec", 0.0), 2),
            "ram_used_percent": metrics.get("ram_used_percent", 0.0),
            "ram_available_gb": metrics.get("ram_available_gb", 0.0)
        }
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    @staticmethod
    def print_progress(
        epoch: int,
        total_epochs: int,
        batch_idx: int,
        total_batches: int,
        critic_loss: float,
        gen_loss: float,
        w_est: float,
        gp: float,
        step_time_sec: float
    ) -> None:
        """Formatted terminal status display."""
        eta_sec = (total_batches - batch_idx) * step_time_sec
        print(
            f"Epoch [{epoch:04d}/{total_epochs:04d}] "
            f"Batch [{batch_idx:03d}/{total_batches:03d}] | "
            f"L_D: {critic_loss:+.4f} | "
            f"L_G: {gen_loss:+.4f} | "
            f"W_est: {w_est:+.4f} | "
            f"GP: {gp:.4f} | "
            f"Batch: {step_time_sec*1000:.0f}ms | "
            f"ETA: {eta_sec:.1f}s",
            end="\r"
        )
