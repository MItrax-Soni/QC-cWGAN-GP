"""
Comparative Synthetic Data Generator.
Extracts class-conditioned synthetic samples from trained VAE, Vanilla cGAN, and cWGAN-GP models
to build dedicated augmented manifests for downstream classification benchmarks.
"""

import os
import sys
import argparse
import pandas as pd
from PIL import Image
import torch
import torchvision.utils as vutils

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.vae import ConditionalVAE
from src.models.cgan import VanillaConditionalDiscriminator
from src.models.generator import ConditionalGenerator
from src.utils.device import get_device


def generate_vae_synthetic_data(
    checkpoint_path: str = r"d:\Projects\GAN\experiments\vae_1000ep\checkpoints\latest.pt",
    output_dir: str = r"d:\Projects\GAN\data\synthetic_vae",
    samples_per_minority_class: int = 50,
    minority_classes: list = None,
    device: torch.device = None
) -> str:
    """Generates synthetic samples from trained Conditional VAE."""
    if device is None:
        device = get_device()
    if minority_classes is None:
        minority_classes = [1, 7]  # contempt, sleepy

    class_names = {1: "contempt", 7: "sleepy"}
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    model = ConditionalVAE(latent_dim=128, num_classes=9).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for cid in minority_classes:
            cname = class_names.get(cid, f"class_{cid}")
            cdir = os.path.join(output_dir, cname)
            os.makedirs(cdir, exist_ok=True)

            labels = torch.full((samples_per_minority_class,), cid, dtype=torch.long, device=device)
            samples = model.sample(labels, device=device)  # in [-1, 1]

            for i in range(samples_per_minority_class):
                fpath = os.path.join(cdir, f"vae_{cname}_{i:04d}.png")
                vutils.save_image(samples[i], fpath, normalize=True, value_range=(-1.0, 1.0))
                manifest_rows.append({
                    "image_path": fpath,
                    "class_id": cid,
                    "class_name": cname,
                    "is_synthetic": True,
                    "model_source": "VAE"
                })

    manifest_csv = os.path.join(output_dir, "synthetic_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"VAE synthetic data manifest generated ({len(manifest_rows)} samples): {manifest_csv}")
    return manifest_csv


def generate_cgan_synthetic_data(
    checkpoint_path: str = r"d:\Projects\GAN\experiments\cgan_1000ep\checkpoints\latest.pt",
    output_dir: str = r"d:\Projects\GAN\data\synthetic_cgan",
    samples_per_minority_class: int = 50,
    minority_classes: list = None,
    device: torch.device = None
) -> str:
    """Generates synthetic samples from trained Vanilla cGAN."""
    if device is None:
        device = get_device()
    if minority_classes is None:
        minority_classes = [1, 7]

    class_names = {1: "contempt", 7: "sleepy"}
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    generator = ConditionalGenerator(latent_dim=100, num_classes=9).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        g_key = "generator_state" if "generator_state" in ckpt else "generator_state_dict"
        generator.load_state_dict(ckpt[g_key])
    generator.eval()

    with torch.no_grad():
        for cid in minority_classes:
            cname = class_names.get(cid, f"class_{cid}")
            cdir = os.path.join(output_dir, cname)
            os.makedirs(cdir, exist_ok=True)

            z = torch.randn(samples_per_minority_class, 100, device=device)
            labels = torch.full((samples_per_minority_class,), cid, dtype=torch.long, device=device)
            samples = generator(z, labels)

            for i in range(samples_per_minority_class):
                fpath = os.path.join(cdir, f"cgan_{cname}_{i:04d}.png")
                vutils.save_image(samples[i], fpath, normalize=True, value_range=(-1.0, 1.0))
                manifest_rows.append({
                    "image_path": fpath,
                    "class_id": cid,
                    "class_name": cname,
                    "is_synthetic": True,
                    "model_source": "cGAN"
                })

    manifest_csv = os.path.join(output_dir, "synthetic_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"cGAN synthetic data manifest generated ({len(manifest_rows)} samples): {manifest_csv}")
    return manifest_csv


def generate_cwgan_unfiltered_data(
    checkpoint_path: str = r"d:\Projects\GAN\experiments\cwgan_gp_1000ep\checkpoints\latest.pt",
    output_dir: str = r"d:\Projects\GAN\data\synthetic_cwgan_unfiltered",
    samples_per_minority_class: int = 50,
    minority_classes: list = None,
    device: torch.device = None
) -> str:
    """Generates raw (unfiltered) synthetic samples from trained cWGAN-GP."""
    if device is None:
        device = get_device()
    if minority_classes is None:
        minority_classes = [1, 7]

    class_names = {1: "contempt", 7: "sleepy"}
    os.makedirs(output_dir, exist_ok=True)
    manifest_rows = []

    generator = ConditionalGenerator(latent_dim=100, num_classes=9).to(device)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        g_key = "generator_state" if "generator_state" in ckpt else "generator_state_dict"
        generator.load_state_dict(ckpt[g_key])
    generator.eval()

    with torch.no_grad():
        for cid in minority_classes:
            cname = class_names.get(cid, f"class_{cid}")
            cdir = os.path.join(output_dir, cname)
            os.makedirs(cdir, exist_ok=True)

            z = torch.randn(samples_per_minority_class, 100, device=device)
            labels = torch.full((samples_per_minority_class,), cid, dtype=torch.long, device=device)
            samples = generator(z, labels)

            for i in range(samples_per_minority_class):
                fpath = os.path.join(cdir, f"cwgan_raw_{cname}_{i:04d}.png")
                vutils.save_image(samples[i], fpath, normalize=True, value_range=(-1.0, 1.0))
                manifest_rows.append({
                    "image_path": fpath,
                    "class_id": cid,
                    "class_name": cname,
                    "is_synthetic": True,
                    "model_source": "cWGAN-GP-Unfiltered"
                })

    manifest_csv = os.path.join(output_dir, "synthetic_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False)
    print(f"cWGAN-GP unfiltered manifest generated ({len(manifest_rows)} samples): {manifest_csv}")
    return manifest_csv
