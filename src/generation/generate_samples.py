"""
Synthetic Sample Generation Module.
Loads a trained cWGAN-GP checkpoint and generates class-conditioned synthetic facial expression samples.
"""

import os
import sys
import argparse
import torch
import torchvision.utils as vutils

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.models.generator import ConditionalGenerator
from src.utils.device import get_device


def generate_samples(
    checkpoint_path: str,
    output_dir: str = r"d:\Projects\GAN\results\generated_samples",
    samples_per_class: int = 8,
    latent_dim: int = 100,
    num_classes: int = 9,
    device: torch.device = None
):
    """
    Generates synthetic samples for each class and saves both individual and grid images.
    """
    if device is None:
        device = get_device()

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading generator from: {checkpoint_path}")
    generator = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        class_embedding_dim=100
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "generator_state_dict" in ckpt:
        generator.load_state_dict(ckpt["generator_state_dict"])
    elif "model_state_dict" in ckpt:
        generator.load_state_dict(ckpt["model_state_dict"])
    else:
        generator.load_state_dict(ckpt)

    generator.eval()

    all_samples = []
    class_names = ["angry", "contempt", "disgust", "fear", "happy", "natural", "sad", "sleepy", "surprised"]

    with torch.no_grad():
        for class_id in range(num_classes):
            c_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            class_dir = os.path.join(output_dir, c_name)
            os.makedirs(class_dir, exist_ok=True)

            z = torch.randn(samples_per_class, latent_dim, device=device)
            labels = torch.full((samples_per_class,), class_id, dtype=torch.long, device=device)

            fakes = generator(z, labels)  # in [-1, 1]
            all_samples.append(fakes)

            for i in range(samples_per_class):
                img_path = os.path.join(class_dir, f"sample_{i:04d}.png")
                # Denormalize to [0, 1]
                vutils.save_image(fakes[i], img_path, normalize=True, value_range=(-1.0, 1.0))

    # Save a master grid across all classes
    all_tensor = torch.cat(all_samples, dim=0)
    grid_path = os.path.join(output_dir, "all_classes_grid.png")
    vutils.save_image(all_tensor, grid_path, nrow=samples_per_class, normalize=True, value_range=(-1.0, 1.0))
    print(f"Sample generation completed. Grid saved to: {grid_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic samples from cWGAN-GP checkpoint.")
    parser.add_argument("--checkpoint", type=str, default=r"d:\Projects\GAN\checkpoints\pilot_5ep\latest.pt", help="Path to checkpoint")
    parser.add_argument("--output_dir", type=str, default=r"d:\Projects\GAN\results\generated_samples", help="Output directory")
    parser.add_argument("--samples_per_class", type=int, default=8, help="Number of samples per class")
    args = parser.parse_args()

    generate_samples(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        samples_per_class=args.samples_per_class
    )
