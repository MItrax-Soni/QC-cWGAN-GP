"""
Conditional Variational Autoencoder (CVAE) Training Engine.
Trains the CVAE baseline model on facial expression crops for comparative analysis.
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
from src.models.vae import ConditionalVAE, compute_vae_loss
from src.data.dataset import get_dataloader
from src.utils.device import get_device


def train_vae(
    manifest_path: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    epochs: int = 1000,
    batch_size: int = 32,
    lr: float = 0.0005,
    latent_dim: int = 128,
    sample_interval: int = 50,
    historical_interval: int = 100,
    output_dir: str = r"d:\Projects\GAN\experiments\vae_1000ep",
    device: torch.device = None
):
    """
    Trains Conditional VAE baseline with crash-proof checkpointing and logging.
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
    print(f"Training Conditional VAE Baseline")
    print(f"Device: {device} | Target Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")
    print(f"=======================================================\n")

    loader = get_dataloader(manifest_path, batch_size=batch_size, shuffle=True)
    model = ConditionalVAE(latent_dim=latent_dim, num_classes=9).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_epoch = 1
    latest_ckpt = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest_ckpt):
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"Resuming VAE from completed Epoch {start_epoch - 1}. Next epoch: {start_epoch}")

    fixed_labels = torch.arange(9, dtype=torch.long, device=device)

    # Initialize CSV if needed
    if not os.path.exists(metrics_csv):
        pd.DataFrame(columns=["epoch", "loss", "recon_loss", "kld", "duration_sec"]).to_csv(metrics_csv, index=False)

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kld = 0.0
        num_batches = 0

        start_time = time.time()
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = model(imgs, lbls)
            loss, recon_val, kld_val = compute_vae_loss(recon, imgs, mu, logvar)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_val
            total_kld += kld_val
            num_batches += 1

        elapsed = time.time() - start_time
        avg_loss = total_loss / max(1, num_batches)
        avg_recon = total_recon / max(1, num_batches)
        avg_kld = total_kld / max(1, num_batches)

        # Log metrics
        row = pd.DataFrame([{
            "epoch": epoch,
            "loss": round(avg_loss, 4),
            "recon_loss": round(avg_recon, 4),
            "kld": round(avg_kld, 4),
            "duration_sec": round(elapsed, 2)
        }])
        row.to_csv(metrics_csv, mode="a", header=False, index=False)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"VAE Epoch [{epoch:04d}/{epochs:04d}] Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}) | Time: {elapsed:.2f}s")

        # Save sample grid
        if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                samples = model.sample(fixed_labels, device=device)
                grid_path = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
                vutils.save_image(samples, grid_path, nrow=3, normalize=True, value_range=(-1.0, 1.0))

        # Atomic checkpoint save
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
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

    print(f"\n[VAE COMPLETE] 1,000-Epoch Training Finished. Model saved at: {latest_ckpt}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Conditional VAE Baseline.")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--manifest", type=str, default=r"d:\Projects\GAN\data\splits\pilot_manifest.csv", help="Manifest path")
    args = parser.parse_args()

    train_vae(manifest_path=args.manifest, epochs=args.epochs, batch_size=args.batch_size)
