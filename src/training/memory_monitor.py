"""
Memory and Health Watchdog Monitor for Long-Running GAN Training.
Tracks system RAM, compute time, and detects numerical instability.
"""

import os
import sys
import time
import ctypes
from typing import Dict, Any
import torch


class MemoryMonitor:
    """
    Monitors RAM, GPU memory (if active), batch timing, and numerical stability.
    """
    def __init__(self, warning_ram_threshold_gb: float = 0.5):
        self.warning_ram_threshold_gb = warning_ram_threshold_gb

    def get_memory_stats(self) -> Dict[str, Any]:
        """Reads system RAM stats via Windows GlobalMemoryStatusEx."""
        stats = {
            "ram_total_gb": 0.0,
            "ram_available_gb": 0.0,
            "ram_used_percent": 0.0,
            "gpu_allocated_mb": 0.0
        }

        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))

            stats["ram_total_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            stats["ram_available_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
            stats["ram_used_percent"] = stat.dwMemoryLoad

        if torch.cuda.is_available():
            stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)

        return stats

    def check_health(self, losses: Dict[str, float]) -> None:
        """
        Validates that losses are real and finite. Raises FloatingPointError if NaN or Inf occurs.
        """
        for k, v in losses.items():
            if torch.is_tensor(v):
                v = v.item()
            if v != v:  # NaN check
                raise FloatingPointError(f"Training Watchdog detected NaN in metric: {k}")
            if abs(v) == float("inf"):
                raise FloatingPointError(f"Training Watchdog detected Inf in metric: {k}")
