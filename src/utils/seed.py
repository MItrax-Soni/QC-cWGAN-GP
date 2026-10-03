"""
Deterministic Random Seed Utility.
Guarantees full reproducibility across Python, NumPy, and PyTorch (CPU and CUDA).
"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Sets random seeds across Python, NumPy, and PyTorch for reproducible runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
