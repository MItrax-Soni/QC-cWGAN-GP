"""
Loss Functions for Wasserstein GAN with Gradient Penalty (cWGAN-GP) and Baselines.
Includes numerical stability monitors to identify NaN and Inf loss values.
"""

from typing import Tuple, Dict
import torch


def compute_critic_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
    gradient_penalty: torch.Tensor,
    lambda_gp: float = 10.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes conditional Critic loss:
    L_D = E[D(fake, y)] - E[D(real, y)] + lambda_gp * GP
    Wasserstein Estimate = E[D(real, y)] - E[D(fake, y)]
    """
    mean_real = real_scores.mean()
    mean_fake = fake_scores.mean()
    wasserstein_estimate = mean_real - mean_fake

    critic_loss = mean_fake - mean_real + lambda_gp * gradient_penalty

    # Guard against NaN/Inf
    if torch.isnan(critic_loss) or torch.isinf(critic_loss):
        raise FloatingPointError(f"Critic loss became non-finite: {critic_loss.item()}")

    metrics = {
        "critic_loss": critic_loss.item(),
        "wasserstein_estimate": wasserstein_estimate.item(),
        "real_score": mean_real.item(),
        "fake_score": mean_fake.item(),
        "gradient_penalty": gradient_penalty.item()
    }
    return critic_loss, metrics


def compute_generator_loss(fake_scores: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes conditional Generator loss:
    L_G = -E[D(fake, y)]
    """
    gen_loss = -fake_scores.mean()

    # Guard against NaN/Inf
    if torch.isnan(gen_loss) or torch.isinf(gen_loss):
        raise FloatingPointError(f"Generator loss became non-finite: {gen_loss.item()}")

    metrics = {
        "generator_loss": gen_loss.item()
    }
    return gen_loss, metrics
