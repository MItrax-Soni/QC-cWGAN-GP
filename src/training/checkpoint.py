"""
Atomic Checkpointing Engine for Fault-Tolerant Extended GAN Training.
Implements crash-proof atomic writes (temp-write + atomic-swap) to guarantee that
checkpoints are never corrupted by unexpected power loss, crashes, or user interrupts.
"""

import os
import random
import numpy as np
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim


def capture_rng_states() -> Dict[str, Any]:
    """Captures all random number generator states for deterministic resumption."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state()
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng_states(states: Dict[str, Any]) -> None:
    """Restores random number generator states."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        np.random.set_state(states["numpy"])
    if "torch_cpu" in states:
        torch.set_rng_state(states["torch_cpu"])
    if "torch_cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def save_checkpoint(
    checkpoint_dir: str,
    epoch: int,
    global_step: int,
    generator: nn.Module,
    critic: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    history: list,
    config: dict,
    is_historical: bool = False,
    extra_data: Optional[dict] = None
) -> str:
    """
    Saves a complete training checkpoint atomically.
    Overwrites 'latest.pt' safely and writes 'epoch_XXXX.pt' for historical milestones.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "generator_state": generator.state_dict(),
        "critic_state": critic.state_dict(),
        "optimizer_g_state": opt_g.state_dict(),
        "optimizer_d_state": opt_d.state_dict(),
        "rng_states": capture_rng_states(),
        "history": history,
        "config": config
    }
    if extra_data:
        state["extra_data"] = extra_data

    # 1. Atomic save for 'latest.pt'
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    temp_path = os.path.join(checkpoint_dir, "latest_tmp.pt")

    # Write to temporary file first
    torch.save(state, temp_path)
    
    # Atomic file rename (POSIX and Windows replace)
    os.replace(temp_path, latest_path)

    # 2. Historical checkpoint (if milestone epoch)
    if is_historical:
        hist_dir = os.path.join(checkpoint_dir, "historical")
        os.makedirs(hist_dir, exist_ok=True)
        hist_path = os.path.join(hist_dir, f"epoch_{epoch:04d}.pt")
        torch.save(state, hist_path)

    return latest_path
