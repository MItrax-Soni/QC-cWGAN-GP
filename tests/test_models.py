"""
Unit Tests for Generative and Discriminative Models.
Validates tensor input/output shapes, parameter gradient flows,
Wasserstein gradient penalty computation, and baseline architectures.
"""

import os
import sys
import unittest
import torch

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.generator import ConditionalGenerator
from src.models.critic import ConditionalCritic
from src.training.gradient_penalty import compute_gradient_penalty
from src.training.losses import compute_critic_loss, compute_generator_loss
from src.models.vae import ConditionalVAE, compute_vae_loss
from src.models.cgan import VanillaConditionalDiscriminator, compute_vanilla_cgan_losses
from src.models.resnet_classifier import ResNetClassifier


class TestModelArchitectures(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.latent_dim = 100
        self.num_classes = 9
        self.image_size = 64

    def test_generator_forward_and_backward(self):
        """Verifies Generator output shape (B, 3, 64, 64), Tanh bounds, and gradients."""
        netG = ConditionalGenerator(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            feature_maps=32
        ).to(self.device)

        z = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        fake_imgs = netG(z, labels)

        self.assertEqual(fake_imgs.shape, (self.batch_size, 3, 64, 64))
        self.assertTrue(torch.all(fake_imgs >= -1.0))
        self.assertTrue(torch.all(fake_imgs <= 1.0))
        self.assertFalse(torch.isnan(fake_imgs).any())

        # Test gradient backprop
        loss = fake_imgs.mean()
        loss.backward()
        for name, param in netG.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")

    def test_critic_forward_and_backward(self):
        """Verifies Critic output shape (B, 1), unconstrained scores, and gradients."""
        netD = ConditionalCritic(
            num_classes=self.num_classes,
            feature_maps=32,
            image_size=self.image_size
        ).to(self.device)

        imgs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        scores = netD(imgs, labels)

        self.assertEqual(scores.shape, (self.batch_size, 1))
        self.assertFalse(torch.isnan(scores).any())
        self.assertFalse(torch.isinf(scores).any())

        # Test gradient backprop
        loss = scores.mean()
        loss.backward()
        for name, param in netD.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")

    def test_gradient_penalty(self):
        """Verifies Gradient Penalty computation, non-negativity, and backprop."""
        netD = ConditionalCritic(
            num_classes=self.num_classes,
            feature_maps=32,
            image_size=self.image_size
        ).to(self.device)

        real_imgs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        fake_imgs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        gp = compute_gradient_penalty(
            critic=netD,
            real_images=real_imgs,
            fake_images=fake_imgs,
            labels=labels,
            device=self.device
        )

        self.assertEqual(gp.dim(), 0)  # Scalar
        self.assertTrue(gp.item() >= 0.0)
        self.assertFalse(torch.isnan(gp))
        self.assertFalse(torch.isinf(gp))

        # Test backpropagation through GP
        gp.backward()
        has_grad = any(p.grad is not None for p in netD.parameters() if p.requires_grad)
        self.assertTrue(has_grad)

    def test_cvae_forward_and_sample(self):
        """Verifies Conditional VAE reconstruction, sampling, and loss computation."""
        vae = ConditionalVAE(
            latent_dim=64,
            num_classes=self.num_classes,
            feature_maps=32,
            image_size=self.image_size
        ).to(self.device)

        imgs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        recon, mu, logvar = vae(imgs, labels)
        self.assertEqual(recon.shape, (self.batch_size, 3, 64, 64))
        self.assertEqual(mu.shape, (self.batch_size, 64))
        self.assertEqual(logvar.shape, (self.batch_size, 64))

        loss, r_loss, kld = compute_vae_loss(recon, imgs, mu, logvar)
        self.assertTrue(loss.item() > 0.0)

        # Test sampling
        samples = vae.sample(labels, device=self.device)
        self.assertEqual(samples.shape, (self.batch_size, 3, 64, 64))

    def test_vanilla_cgan(self):
        """Verifies Vanilla cGAN Discriminator Sigmoid output and loss computation."""
        netG = ConditionalGenerator(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            feature_maps=32
        ).to(self.device)
        netD = VanillaConditionalDiscriminator(
            num_classes=self.num_classes,
            feature_maps=32,
            image_size=self.image_size
        ).to(self.device)

        real_imgs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)
        labels = torch.randint(0, self.num_classes, (self.batch_size,), device=self.device)

        probs = netD(real_imgs, labels)
        self.assertEqual(probs.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))

        d_loss, g_loss, metrics = compute_vanilla_cgan_losses(
            discriminator=netD,
            generator=netG,
            real_images=real_imgs,
            labels=labels,
            latent_dim=self.latent_dim,
            device=self.device
        )
        self.assertTrue(d_loss.item() > 0.0)
        self.assertTrue(g_loss.item() > 0.0)

    def test_resnet_classifier(self):
        """Verifies ResNet-18 logits shape (B, 9) and feature extraction shape (B, 512)."""
        model = ResNetClassifier(num_classes=self.num_classes, pretrained=False).to(self.device)
        imgs = torch.randn(self.batch_size, 3, 64, 64, device=self.device)

        logits = model(imgs)
        self.assertEqual(logits.shape, (self.batch_size, 9))

        features = model.extract_features(imgs)
        self.assertEqual(features.shape, (self.batch_size, 512))
        self.assertFalse(torch.isnan(features).any())


if __name__ == "__main__":
    unittest.main()
