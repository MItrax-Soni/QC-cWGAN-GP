"""
Conditional Critic Architecture for Conditional WGAN-GP.
Outputs an unconstrained scalar Wasserstein score estimating E[D(x, y)]
without applying Sigmoid or Batch Normalization.
"""

import torch
import torch.nn as nn
from src.models.initialization import weights_init_normal


class ConditionalCritic(nn.Module):
    """
    Conditional Critic (Discriminator) network estimating the Wasserstein score.
    Uses Instance Normalization instead of Batch Normalization to preserve sample-independent
    Lipschitz continuity required by the Gradient Penalty.
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

        # Class embedding and spatial projection
        self.label_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.label_projector = nn.Linear(class_embedding_dim, image_size * image_size)

        # Total input channels = RGB channels (3) + Condition channel (1)
        in_channels = channels + 1

        # Downsampling path: (64x64) -> (32x32) -> (16x16) -> (8x8) -> (4x4)
        self.conv_blocks = nn.Sequential(
            # Block 1: (4, 64, 64) -> (feature_maps, 32, 32) - No norm on first layer
            nn.Conv2d(in_channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (feature_maps, 32, 32) -> (feature_maps * 2, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (feature_maps * 2, 16, 16) -> (feature_maps * 4, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (feature_maps * 4, 8, 8) -> (feature_maps * 8, 4, 4)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(feature_maps * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 5: (feature_maps * 8, 4, 4) -> (1, 1, 1) scalar score output
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=True)
        )

        self.apply(weights_init_normal)

    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            img: Image tensor of shape (batch_size, 3, 64, 64).
            labels: Long tensor of shape (batch_size,).
        Returns:
            Scalar Wasserstein score tensor of shape (batch_size, 1).
        """
        batch_size, _, h, w = img.shape

        # Embed and spatially project class label to (batch_size, 1, 64, 64)
        c = self.label_embedding(labels)
        c_proj = self.label_projector(c).view(batch_size, 1, h, w)

        # Concatenate along channel dimension
        x = torch.cat([img, c_proj], dim=1)

        # Compute Wasserstein score
        score = self.conv_blocks(x)
        return score.view(batch_size, 1)
