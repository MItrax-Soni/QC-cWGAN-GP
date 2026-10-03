"""
Device Management and Hardware Detection Utility.
Provides device selection and hardware reporting.
"""

import os
import torch


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Returns the appropriate PyTorch device.
    Defaults to CPU for systems without supported CUDA architectures (e.g. Kepler GT 710).
    """
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict:
    """
    Returns diagnostic information about the active compute device.
    """
    info = {"device_type": device.type}
    if device.type == "cuda":
        info["device_name"] = torch.cuda.get_device_name(device)
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2)
        info["vram_allocated_gb"] = round(torch.cuda.memory_allocated(device) / (1024**3), 2)
    else:
        info["device_name"] = "CPU (Host Processor)"
    return info
