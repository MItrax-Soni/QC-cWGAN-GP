"""
Gradient Penalty Implementation for Wasserstein GAN (WGAN-GP).
Enforces the 1-Lipschitz continuity constraint via sample-wise gradient norm penalty:
E[(||nabla_{hat{x}} D(hat{x}, y)||_2 - 1)^2].
"""

import torch
import torch.nn as nn


def compute_gradient_penalty(
    critic: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Computes the WGAN-GP gradient penalty for conditional critic.
    Args:
        critic: ConditionalCritic module.
        real_images: Real batch tensor of shape (B, C, H, W).
        fake_images: Generated batch tensor of shape (B, C, H, W) (must be detached).
        labels: Class labels tensor of shape (B,).
        device: Active compute device.
    Returns:
        Scalar gradient penalty tensor.
    """
    batch_size = real_images.size(0)

    # Sample random interpolation coefficient epsilon ~ U[0, 1] per sample
    # Shape: (B, 1, 1, 1) for proper spatial broadcasting
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device, dtype=real_images.dtype)

    # Compute interpolated images hat{x} = epsilon * real + (1 - epsilon) * fake
    interpolated = (epsilon * real_images + (1.0 - epsilon) * fake_images).requires_grad_(True)

    # Forward pass through critic with conditioned labels
    critic_scores = critic(interpolated, labels)

    # Create gradient outputs vector of ones with identical shape as scores
    grad_outputs = torch.ones_like(critic_scores, requires_grad=False, device=device)

    # Calculate gradients of scores with respect to interpolated inputs
    gradients = torch.autograd.grad(
        outputs=critic_scores,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Flatten gradients to (B, -1) to compute norm per individual sample
    gradients = gradients.view(batch_size, -1)

    # Compute per-sample L2 norm with small epsilon for numerical stability
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty = Mean over batch of (||grad||_2 - 1)^2
    gradient_penalty = torch.mean((gradient_norm - 1.0) ** 2)

    # Check for NaN / Inf
    if torch.isnan(gradient_penalty) or torch.isinf(gradient_penalty):
        raise FloatingPointError("Gradient Penalty computation produced NaN or Inf!")

    return gradient_penalty
