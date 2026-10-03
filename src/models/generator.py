"""
Conditional Generator Architecture for cWGAN-GP and cGAN.
Synthesizes 64x64 RGB facial expression images conditioned on emotion class labels.
"""

import torch
import torch.nn as nn
from src.models.initialization import weights_init_normal


class ConditionalGenerator(nn.Module):
    """
    Conditional Generator network mapping random noise vector z and emotion class label y
    to a synthetic facial image of shape (batch_size, 3, 64, 64).
    """
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 9,
        class_embedding_dim: int = 100,
        channels: int = 3,
        feature_maps: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.channels = channels
        self.feature_maps = feature_maps

        # Learned class embedding
        self.label_embedding = nn.Embedding(num_classes, class_embedding_dim)

        # Input dimension = noise dimension + class embedding dimension
        input_dim = latent_dim + class_embedding_dim

        # Initial linear projection to 4x4 spatial feature map
        self.fc = nn.Sequential(
            nn.Linear(input_dim, feature_maps * 8 * 4 * 4),
            nn.BatchNorm1d(feature_maps * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Transposed convolution upsampling blocks: (4x4) -> (8x8) -> (16x16) -> (32x32) -> (64x64)
        self.deconv_blocks = nn.Sequential(
            # (feature_maps * 8, 4, 4) -> (feature_maps * 4, 8, 8)
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # (feature_maps * 4, 8, 8) -> (feature_maps * 2, 16, 16)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # (feature_maps * 2, 16, 16) -> (feature_maps, 32, 32)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # (feature_maps, 32, 32) -> (channels, 64, 64)
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

        self.apply(weights_init_normal)

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            z: Latent noise tensor of shape (batch_size, latent_dim).
            labels: Long tensor of class labels of shape (batch_size,).
        Returns:
            Synthetic images of shape (batch_size, 3, 64, 64) with pixel values in [-1, 1].
        """
        # Embed emotion labels: (batch_size, class_embedding_dim)
        c = self.label_embedding(labels)

        # Concatenate noise and class condition
        x = torch.cat([z, c], dim=1)

        # Project and reshape to feature maps
        x = self.fc(x)
        x = x.view(x.size(0), self.feature_maps * 8, 4, 4)

        # Upsample to 64x64
        img = self.deconv_blocks(x)
        return img
