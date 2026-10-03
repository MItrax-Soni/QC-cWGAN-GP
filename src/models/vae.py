"""
Conditional Variational Autoencoder (CVAE) Baseline Architecture.
Encodes 64x64 facial expression images conditioned on emotion labels and synthesizes
faces via Gaussian latent sampling.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalVAE(nn.Module):
    """
    Conditional VAE baseline with convolutional encoder, reparameterization trick,
    and transposed convolutional decoder for 64x64 images.
    """
    def __init__(
        self,
        latent_dim: int = 128,
        num_classes: int = 9,
        class_embedding_dim: int = 64,
        channels: int = 3,
        feature_maps: int = 64,
        image_size: int = 64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.class_embedding_dim = class_embedding_dim
        self.channels = channels
        self.feature_maps = feature_maps
        self.image_size = image_size

        # Class embedding for encoder and decoder
        self.label_embedding = nn.Embedding(num_classes, class_embedding_dim)
        self.enc_label_proj = nn.Linear(class_embedding_dim, image_size * image_size)

        # Encoder: (4, 64, 64) -> (feature_maps * 8, 4, 4)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(channels + 1, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True)
        )

        flat_dim = feature_maps * 8 * 4 * 4
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)

        # Decoder: (latent_dim + class_embedding_dim) -> (3, 64, 64)
        self.dec_fc = nn.Sequential(
            nn.Linear(latent_dim + class_embedding_dim, flat_dim),
            nn.BatchNorm1d(flat_dim),
            nn.ReLU(True)
        )

        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps, eps ~ N(0, 1)."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, h, w = x.shape
        c = self.label_embedding(labels)
        c_proj = self.enc_label_proj(c).view(batch_size, 1, h, w)
        inp = torch.cat([x, c_proj], dim=1)
        h_enc = self.encoder_conv(inp).view(batch_size, -1)
        mu = self.fc_mu(h_enc)
        logvar = self.fc_logvar(h_enc)
        return mu, logvar

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        c = self.label_embedding(labels)
        inp = torch.cat([z, c], dim=1)
        h_dec = self.dec_fc(inp).view(batch_size, self.feature_maps * 8, 4, 4)
        out = self.decoder_deconv(h_dec)
        return out

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar

    def sample(self, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Samples synthetic facial images from standard prior z ~ N(0, I)."""
        batch_size = labels.size(0)
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.decode(z, labels.to(device))


def compute_vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0
) -> Tuple[torch.Tensor, float, float]:
    """
    Computes VAE loss = Reconstruction MSE + beta * KL Divergence.
    """
    recon_loss = F.mse_loss(recon, target, reduction="sum") / target.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / target.size(0)
    total_loss = recon_loss + beta * kld
    return total_loss, recon_loss.item(), kld.item()
