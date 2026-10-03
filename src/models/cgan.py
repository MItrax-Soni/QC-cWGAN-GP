"""
Vanilla Conditional GAN (cGAN) Baseline Architecture.
Implements the standard minimax GAN objective with Binary Cross-Entropy loss
and Sigmoid discriminator probability output.
"""

from typing import Tuple, Dict
import torch
import torch.nn as nn
from src.models.generator import ConditionalGenerator
from src.models.initialization import weights_init_normal


class VanillaConditionalDiscriminator(nn.Module):
    """
    Conditional Discriminator with Sigmoid activation outputting probability D(x, y) in [0, 1].
    """
    def __init__(
        self,
        num_classes: int = 9,
        class_embedding_dim: int = 100,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64
    ):
        super().__init__()
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.channels = channels
        self.feature_maps = feature_maps
        self.image_size = image_size

        self.label_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.label_projector = nn.Linear(class_embedding_dim, image_size * image_size)

        self.conv_blocks = nn.Sequential(
            # (4, 64, 64) -> (feature_maps, 32, 32)
            nn.Conv2d(channels + 1, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps, 32, 32) -> (feature_maps * 2, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps * 2, 16, 16) -> (feature_maps * 4, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps * 4, 8, 8) -> (feature_maps * 8, 4, 4)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_maps * 8, 4, 4) -> (1, 1, 1) -> Sigmoid
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.apply(weights_init_normal)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size, _, h, w = img.shape
        c = self.label_embedding(labels)
        c_proj = self.label_projector(c).view(batch_size, 1, h, w)
        x = torch.cat([img, c_proj], dim=1)
        prob = self.conv_blocks(x)
        return prob.view(batch_size, 1)


def compute_vanilla_cgan_losses(
    discriminator: nn.Module,
    generator: nn.Module,
    real_images: torch.Tensor,
    labels: torch.Tensor,
    latent_dim: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Computes standard Minimax BCE losses for Vanilla cGAN.
    """
    batch_size = real_images.size(0)
    bce_loss = nn.BCELoss()

    real_targets = torch.ones(batch_size, 1, device=device)
    fake_targets = torch.zeros(batch_size, 1, device=device)

    # 1. Train Discriminator
    real_preds = discriminator(real_images, labels)
    d_real_loss = bce_loss(real_preds, real_targets)

    z = torch.randn(batch_size, latent_dim, device=device)
    fake_images = generator(z, labels).detach()
    fake_preds = discriminator(fake_images, labels)
    d_fake_loss = bce_loss(fake_preds, fake_targets)

    d_loss = (d_real_loss + d_fake_loss) * 0.5

    # 2. Train Generator
    z2 = torch.randn(batch_size, latent_dim, device=device)
    gen_images = generator(z2, labels)
    gen_preds = discriminator(gen_images, labels)
    g_loss = bce_loss(gen_preds, real_targets)

    metrics = {
        "d_loss": d_loss.item(),
        "g_loss": g_loss.item(),
        "d_real": real_preds.mean().item(),
        "d_fake": fake_preds.mean().item()
    }
    return d_loss, g_loss, metrics
