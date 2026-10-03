"""
Fixed Noise Generator and Visual Progress Tracker.
Preserves identical latent vectors across epochs to reliably monitor visual convergence,
facial feature sharpness, and emotion expression distinctiveness over 1000 epochs.
"""

import os
import torch
import torchvision.utils as vutils
from typing import Tuple

from src.data.preprocessing import denormalize_tensor
from src.data.yolo_parser import id_to_class_name


def get_fixed_evaluation_batch(
    num_classes: int = 9,
    samples_per_class: int = 4,
    latent_dim: int = 100,
    seed: int = 42,
    device: torch.device = torch.device("cpu")
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates a deterministic set of latent noise vectors and labels for quality tracking.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    total_samples = num_classes * samples_per_class
    # Shape: (total_samples, latent_dim)
    fixed_z = torch.randn(total_samples, latent_dim, generator=generator, device=device)
    
    # Class labels: [0, 0, 0, 0, 1, 1, 1, 1, ..., 8, 8, 8, 8]
    fixed_labels = torch.tensor(
        [cls for cls in range(num_classes) for _ in range(samples_per_class)],
        dtype=torch.long,
        device=device
    )
    return fixed_z, fixed_labels


def save_sample_grid(
    generator: torch.nn.Module,
    fixed_z: torch.Tensor,
    fixed_labels: torch.Tensor,
    output_path: str,
    num_classes: int = 9,
    samples_per_class: int = 4,
    nrow: int = 9
) -> None:
    """
    Evaluates generator on fixed noise batch and saves a tiled image grid.
    """
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_z, fixed_labels)
        # Denormalize from [-1, 1] to [0, 1]
        denorm_imgs = denormalize_tensor(fake_imgs)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        vutils.save_image(
            denorm_imgs,
            output_path,
            nrow=nrow,
            normalize=False,
            padding=2
        )
    generator.train()
