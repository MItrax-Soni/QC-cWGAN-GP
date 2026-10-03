"""
Vanilla Conditional GAN (cGAN) Training Engine.
Trains the Minimax BCE baseline cGAN on facial expression crops for comparative analysis.
Includes atomic checkpointing, resume support, CSV metrics logging, and visual progression grids.
"""

import os
import sys
import argparse
import time
import pandas as pd
import torch
import torch.optim as optim
import torchvision.utils as vutils

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.generator import ConditionalGenerator
from src.models.cgan import VanillaConditionalDiscriminator, compute_vanilla_cgan_losses
from src.data.dataset import get_dataloader
from src.utils.device import get_device


def train_cgan(
    manifest_path: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 0.0002,
    latent_dim: int = 100,
    sample_interval: int = 50,
    historical_interval: int = 100,
    output_dir: str = r"d:\Projects\GAN\experiments\cgan_1000ep",
    device: torch.device = None
):
    """
    Trains Vanilla Conditional GAN baseline with crash-proof checkpointing and logging.
    """
    if device is None:
        device = get_device()

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    samples_dir = os.path.join(output_dir, "samples")
    hist_dir = os.path.join(ckpt_dir, "historical")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(hist_dir, exist_ok=True)

    metrics_csv = os.path.join(output_dir, "metrics.csv")

    print(f"\n=======================================================")
    print(f"Training Vanilla Conditional GAN Baseline")
    print(f"Device: {device} | Target Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"=======================================================\n")

    loader = get_dataloader(manifest_path, batch_size=batch_size, shuffle=True)
    generator = ConditionalGenerator(latent_dim=latent_dim, num_classes=9).to(device)
    discriminator = VanillaConditionalDiscriminator(num_classes=9).to(device)

    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    start_epoch = 1
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        generator.load_state_dict(ckpt["generator_state_dict"])
        discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        if "opt_g_state_dict" in ckpt:
            opt_g.load_state_dict(ckpt["opt_g_state_dict"])
        if "opt_d_state_dict" in ckpt:
            opt_d.load_state_dict(ckpt["opt_d_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming cGAN from completed Epoch {start_epoch - 1}. Next epoch: {start_epoch}")

    fixed_z = torch.randn(9, latent_dim, device=device)
    fixed_labels = torch.arange(9, dtype=torch.long, device=device)

    if not os.path.exists(metrics_csv):
        pd.DataFrame(columns=["epoch", "d_loss", "g_loss", "d_real", "d_fake", "duration_sec"]).to_csv(metrics_csv, index=False)

    for epoch in range(start_epoch, epochs + 1):
        generator.train()
        discriminator.train()
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_d_real = 0.0
        total_d_fake = 0.0
        num_batches = 0

        start_time = time.time()
        bce_loss = torch.nn.BCELoss()
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)
            batch_sz = imgs.size(0)

            real_targets = torch.ones(batch_sz, 1, device=device)
            fake_targets = torch.zeros(batch_sz, 1, device=device)

            # 1. Update Discriminator
            opt_d.zero_grad()
            real_preds = discriminator(imgs, lbls)
            d_real_loss = bce_loss(real_preds, real_targets)

            z = torch.randn(batch_sz, latent_dim, device=device)
            fake_images = generator(z, lbls).detach()
            fake_preds = discriminator(fake_images, lbls)
            d_fake_loss = bce_loss(fake_preds, fake_targets)

            d_loss = 0.5 * (d_real_loss + d_fake_loss)
            d_loss.backward()
            opt_d.step()

            # 2. Update Generator
            opt_g.zero_grad()
            z2 = torch.randn(batch_sz, latent_dim, device=device)
            gen_images = generator(z2, lbls)
            gen_preds = discriminator(gen_images, lbls)
            g_loss = bce_loss(gen_preds, real_targets)
            g_loss.backward()
            opt_g.step()

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            total_d_real += real_preds.mean().item()
            total_d_fake += fake_preds.mean().item()
            num_batches += 1

        elapsed = time.time() - start_time
        avg_d = total_d_loss / max(1, num_batches)
        avg_g = total_g_loss / max(1, num_batches)
        avg_real = total_d_real / max(1, num_batches)
        avg_fake = total_d_fake / max(1, num_batches)

        # Log row
        row = pd.DataFrame([{
            "epoch": epoch,
            "d_loss": round(avg_d, 4),
            "g_loss": round(avg_g, 4),
            "d_real": round(avg_real, 4),
            "d_fake": round(avg_fake, 4),
            "duration_sec": round(elapsed, 2)
        }])
        row.to_csv(metrics_csv, mode="a", header=False, index=False)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"cGAN Epoch [{epoch:04d}/{epochs:04d}] D Loss: {avg_d:.4f} | G Loss: {avg_g:.4f} (Real: {avg_real:.2f}, Fake: {avg_fake:.2f}) | Time: {elapsed:.2f}s")

        # Save sample grid
        if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_z, fixed_labels)
                grid_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
                vutils.save_image(samples, grid_path, nrow=3, normalize=True, value_range=(-1.0, 1.0))

        # Atomic checkpoint save
        ckpt_data = {
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "opt_g_state_dict": opt_g.state_dict(),
            "opt_d_state_dict": opt_d.state_dict()
        }
        tmp_ckpt = latest_ckpt + ".tmp"
        torch.save(ckpt_data, tmp_ckpt)
        if os.path.exists(latest_ckpt):
            os.remove(latest_ckpt)
        os.rename(tmp_ckpt, latest_ckpt)

        # Historical checkpoint
        if epoch % historical_interval == 0 or epoch == epochs:
            hist_path = os.path.join(hist_dir, f"epoch_{epoch:04d}.pt")
            torch.save(ckpt_data, hist_path)

    print(f"\n[cGAN COMPLETE] 1,000-Epoch Training Finished. Model saved at: {latest_ckpt}")
    return generator, discriminator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Vanilla Conditional GAN Baseline.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--manifest", type=str, default=r"d:\Projects\GAN\data\splits\pilot_manifest.csv", help="Manifest path")
    args = parser.parse_args()

    train_cgan(manifest_path=args.manifest, epochs=args.epochs, batch_size=args.batch_size)
