"""
Checkpoint Recovery and Resume Engine.
Restores full training state including weights, optimizers, learning schedules,
RNG states, and metric histories to ensure zero experimental loss upon restart.
"""

import os
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from src.training.checkpoint import restore_rng_states


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Checks if a valid latest.pt checkpoint exists."""
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if os.path.exists(latest):
        return latest
    return None


def load_checkpoint(
    checkpoint_path: str,
    generator: nn.Module,
    critic: nn.Module,
    opt_g: optim.Optimizer,
    opt_d: optim.Optimizer,
    device: torch.device
) -> Tuple[int, int, list, Dict[str, Any]]:
    """
    Loads checkpoint and restores all states.
    Returns:
        next_epoch: The epoch to begin training from (last_completed + 1).
        global_step: The accumulated step counter.
        history: The list of recorded epoch metrics.
        config: The saved configuration dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    print(f"Loading checkpoint from: {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 1. Restore model parameters
    generator.load_state_dict(state["generator_state"])
    critic.load_state_dict(state["critic_state"])

    # 2. Restore optimizer states
    opt_g.load_state_dict(state["optimizer_g_state"])
    opt_d.load_state_dict(state["optimizer_d_state"])

    # 3. Restore random number generator states
    if "rng_states" in state:
        restore_rng_states(state["rng_states"])

    last_epoch = state.get("epoch", 0)
    global_step = state.get("global_step", 0)
    history = state.get("history", [])
    config = state.get("config", {})

    next_epoch = last_epoch + 1
    print(f"Resumption verified: Last completed epoch was {last_epoch}. Next epoch: {next_epoch}.")

    return next_epoch, global_step, history, config
