"""
Weight Initialization Utilities for Generative and Discriminative Models.
Applies normal distribution N(0, 0.02) to Conv and Linear weights as standard
in DCGAN and WGAN-GP literature.
"""

import torch
import torch.nn as nn


def weights_init_normal(m: nn.Module):
    """
    Applies Gaussian initialization to convolutional, linear, and normalization layers.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
