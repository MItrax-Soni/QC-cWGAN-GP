"""
Configuration Loader and Parser.
Reads YAML configuration files and returns strongly-typed dictionaries.
"""

import os
import yaml
from typing import Dict, Any


def load_config(config_path: str = r"d:\Projects\GAN\configs\config.yaml") -> Dict[str, Any]:
    """
    Loads YAML project configuration file safely.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
