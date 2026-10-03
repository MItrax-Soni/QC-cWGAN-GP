"""
Core Training Engine for Conditional Wasserstein GAN with Gradient Penalty (cWGAN-GP).
Implements the 5-to-1 critic/generator ratio, gradient penalty, crash-proof checkpointing,
graceful interruption handling, and deterministic sample progression logging.
"""

import os
import sys
import time
import signal
from typing import Optional, Dict, Any
import yaml
import torch
import torch.optim as optim

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.device import get_device, get_device_info
from src.utils.logger import ExperimentLogger
from src.models.generator import ConditionalGenerator
from src.models.critic import ConditionalCritic
from src.training.gradient_penalty import compute_gradient_penalty
from src.training.losses import compute_critic_loss, compute_generator_loss
from src.training.checkpoint import save_checkpoint
from src.training.resume import find_latest_checkpoint, load_checkpoint
from src.training.memory_monitor import MemoryMonitor
from src.generation.fixed_noise import get_fixed_evaluation_batch, save_sample_grid
from src.data.dataset import get_dataloader


class GracefulKiller:
    """Catches SIGINT (Ctrl+C) and SIGTERM to trigger graceful checkpointing and shutdown."""
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n[WATCHDOG ALERT] Interruption signal received. Completing safe shutdown...")
        self.kill_now = True


def train_cwgan_gp(
    manifest_path: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    zip_path: str = r"C:\Users\mitra\Downloads\archive.zip",
    config_path: str = r"d:\Projects\GAN\configs\config.yaml",
    experiment_id: str = "pilot_run",
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    resume: bool = True,
    sample_interval: int = 5,
    checkpoint_interval: int = 1,
    historical_interval: int = 10,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Executes conditional WGAN-GP training.
    """
    # 1. Load Configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    epochs = epochs or cfg["training"]["target_epochs"]
    batch_size = batch_size or cfg["training"]["batch_size"]
    lr_g = cfg["training"]["lr_g"]
    lr_d = cfg["training"]["lr_d"]
    beta1 = cfg["training"]["beta1"]
    beta2 = cfg["training"]["beta2"]
    critic_iters = cfg["training"]["critic_iterations"]
    lambda_gp = cfg["training"]["lambda_gp"]
    latent_dim = cfg["models"]["latent_dim"]
    num_classes = cfg["dataset"]["num_classes"]
    class_embedding_dim = cfg["models"]["class_embedding_dim"]
    img_size = cfg["dataset"]["image_size"]
    sample_interval = cfg["training"].get("sample_generation_interval", 10)
    historical_interval = cfg["training"].get("history_checkpoint_interval", 25)

    if device is None:
        device = get_device(force_cpu=not cfg["hardware"].get("allow_cuda", False))

    print(f"\n=======================================================")
    print(f"Starting cWGAN-GP Experiment: '{experiment_id}'")
    print(f"Device: {device} | Target Epochs: {epochs} | Batch Size: {batch_size}")
    print(f"Critic Iterations per Gen: {critic_iters} | Lambda GP: {lambda_gp}")
    print(f"=======================================================\n")

    # 2. Setup Directories and Logging
    exp_dir = os.path.join(r"d:\Projects\GAN\experiments", experiment_id)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    samples_dir = os.path.join(exp_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    logger = ExperimentLogger(experiment_dir=exp_dir, config=cfg)
    memory_monitor = MemoryMonitor()
    killer = GracefulKiller()

    # 3. Setup Models & Optimizers
    netG = ConditionalGenerator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        class_embedding_dim=class_embedding_dim,
        feature_maps=cfg["models"]["generator"]["feature_maps"]
    ).to(device)

    netD = ConditionalCritic(
        num_classes=num_classes,
        class_embedding_dim=class_embedding_dim,
        feature_maps=cfg["models"]["critic"]["feature_maps"],
        image_size=img_size
    ).to(device)

    optG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1, beta2))
    optD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1, beta2))

    # 4. Fixed Evaluation Batch for Progress Tracking
    fixed_z, fixed_labels = get_fixed_evaluation_batch(
        num_classes=num_classes,
        samples_per_class=4,
        latent_dim=latent_dim,
        seed=cfg["project"]["seed"],
        device=device
    )

    # 5. Checkpoint Resume Logic
    start_epoch = 1
    global_step = 0
    history = []

    if resume:
        latest_ckpt = find_latest_checkpoint(ckpt_dir)
        if latest_ckpt:
            start_epoch, global_step, history, _ = load_checkpoint(
                latest_ckpt, netG, netD, optG, optD, device
            )
            if start_epoch > epochs:
                print(f"Experiment already completed up to epoch {start_epoch - 1}. No further epochs required.")
                return {"completed": True, "last_epoch": start_epoch - 1, "history": history}

    # 6. Data Loader
    dataloader = get_dataloader(
        manifest=manifest_path,
        zip_path=zip_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        image_size=img_size,
        drop_last=True
    )
    total_batches = len(dataloader)
    print(f"DataLoader initialized: {len(dataloader.dataset)} samples -> {total_batches} batches/epoch.")

    # 7. Training Loop
    for epoch in range(start_epoch, epochs + 1):
        if killer.kill_now:
            break

        epoch_start_time = time.time()
        running_d_loss = 0.0
        running_g_loss = 0.0
        running_w_est = 0.0
        running_gp = 0.0
        running_real_score = 0.0
        running_fake_score = 0.0
        g_updates = 0

        data_iter = iter(dataloader)
        batch_idx = 0

        while batch_idx < total_batches:
            if killer.kill_now:
                break

            step_start = time.time()

            # -----------------------------------------------------------
            # (1) Update Critic (5 iterations)
            # -----------------------------------------------------------
            for _ in range(critic_iters):
                try:
                    real_imgs, real_labels = next(data_iter)
                    batch_idx += 1
                except StopIteration:
                    break

                real_imgs = real_imgs.to(device)
                real_labels = real_labels.to(device)
                b_size = real_imgs.size(0)

                optD.zero_grad()

                # Generate fake images from random noise
                z = torch.randn(b_size, latent_dim, device=device)
                fake_imgs = netG(z, real_labels).detach()

                # Calculate critic scores
                real_scores = netD(real_imgs, real_labels)
                fake_scores = netD(fake_imgs, real_labels)

                # Gradient penalty
                gp = compute_gradient_penalty(
                    critic=netD,
                    real_images=real_imgs,
                    fake_images=fake_imgs,
                    labels=real_labels,
                    device=device
                )

                d_loss, metrics_d = compute_critic_loss(
                    real_scores=real_scores,
                    fake_scores=fake_scores,
                    gradient_penalty=gp,
                    lambda_gp=lambda_gp
                )

                d_loss.backward()
                optD.step()

                running_d_loss += metrics_d["critic_loss"]
                running_w_est += metrics_d["wasserstein_estimate"]
                running_gp += metrics_d["gradient_penalty"]
                running_real_score += metrics_d["real_score"]
                running_fake_score += metrics_d["fake_score"]
                global_step += 1

            # -----------------------------------------------------------
            # (2) Update Generator (1 iteration)
            # -----------------------------------------------------------
            optG.zero_grad()

            z_gen = torch.randn(batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (batch_size,), device=device)

            gen_fake_imgs = netG(z_gen, gen_labels)
            gen_fake_scores = netD(gen_fake_imgs, gen_labels)

            g_loss, metrics_g = compute_generator_loss(gen_fake_scores)
            g_loss.backward()
            optG.step()

            running_g_loss += metrics_g["generator_loss"]
            g_updates += 1

            step_time = time.time() - step_start
            logger.print_progress(
                epoch=epoch,
                total_epochs=epochs,
                batch_idx=batch_idx,
                total_batches=total_batches,
                critic_loss=metrics_d["critic_loss"],
                gen_loss=metrics_g["generator_loss"],
                w_est=metrics_d["wasserstein_estimate"],
                gp=metrics_d["gradient_penalty"],
                step_time_sec=step_time
            )

        print()  # Newline after epoch progress

        # -----------------------------------------------------------
        # Epoch Summary & Logging
        # -----------------------------------------------------------
        epoch_dur = time.time() - epoch_start_time
        num_d_steps = max(1, batch_idx)
        num_g_steps = max(1, g_updates)

        epoch_metrics = {
            "global_step": global_step,
            "critic_loss": running_d_loss / num_d_steps,
            "generator_loss": running_g_loss / num_g_steps,
            "wasserstein_estimate": running_w_est / num_d_steps,
            "gradient_penalty": running_gp / num_d_steps,
            "real_score": running_real_score / num_d_steps,
            "fake_score": running_fake_score / num_d_steps,
            "epoch_duration_sec": epoch_dur
        }

        # Memory diagnostics
        mem_stats = memory_monitor.get_memory_stats()
        epoch_metrics.update(mem_stats)
        memory_monitor.check_health(epoch_metrics)

        history.append(epoch_metrics)
        logger.log_epoch(epoch, epoch_metrics)

        print(
            f"Epoch {epoch:04d} Finished | Duration: {epoch_dur:.1f}s | "
            f"Critic Loss: {epoch_metrics['critic_loss']:+.4f} | "
            f"Gen Loss: {epoch_metrics['generator_loss']:+.4f} | "
            f"W-Dist: {epoch_metrics['wasserstein_estimate']:+.4f} | "
            f"RAM: {mem_stats['ram_used_percent']}%"
        )

        # -----------------------------------------------------------
        # Visual Progress Generation
        # -----------------------------------------------------------
        if epoch == 1 or epoch % sample_interval == 0 or epoch == epochs:
            grid_out = os.path.join(samples_dir, f"epoch_{epoch:04d}.png")
            save_sample_grid(
                generator=netG,
                fixed_z=fixed_z,
                fixed_labels=fixed_labels,
                output_path=grid_out,
                num_classes=num_classes,
                samples_per_class=4
            )
            print(f"Sample visual progression saved: {grid_out}")

        # -----------------------------------------------------------
        # Atomic Checkpointing
        # -----------------------------------------------------------
        is_hist = (epoch % historical_interval == 0) or (epoch == epochs)
        save_checkpoint(
            checkpoint_dir=ckpt_dir,
            epoch=epoch,
            global_step=global_step,
            generator=netG,
            critic=netD,
            opt_g=optG,
            opt_d=optD,
            history=history,
            config=cfg,
            is_historical=is_hist
        )

    # Clean exit
    if killer.kill_now:
        print("[SHUTDOWN] Process safely halted. Latest checkpoint and all logs successfully preserved.")

    return {
        "completed": not killer.kill_now,
        "last_epoch": epoch if 'epoch' in locals() else start_epoch - 1,
        "history": history
    }
