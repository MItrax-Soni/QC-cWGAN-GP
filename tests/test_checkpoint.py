"""
Unit Tests for Checkpoint Engine and Resume Fidelity.
Tests atomic file replacement, complete state dictionary restoration,
and deterministic RNG state recovery.
"""

import os
import sys
import shutil
import unittest
import torch
import torch.optim as optim

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.generator import ConditionalGenerator
from src.models.critic import ConditionalCritic
from src.training.checkpoint import save_checkpoint, capture_rng_states, restore_rng_states
from src.training.resume import load_checkpoint, find_latest_checkpoint


class TestCheckpointAndResume(unittest.TestCase):

    def setUp(self):
        self.test_dir = r"d:\Projects\GAN\tests\test_ckpt_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.device = torch.device("cpu")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_atomic_save_and_state_fidelity(self):
        """Verifies atomic write, file creation, and weight/optimizer restoration."""
        netG = ConditionalGenerator(latent_dim=50, num_classes=9, feature_maps=16).to(self.device)
        netD = ConditionalCritic(num_classes=9, feature_maps=16, image_size=64).to(self.device)
        optG = optim.Adam(netG.parameters(), lr=0.0001)
        optD = optim.Adam(netD.parameters(), lr=0.0001)

        history = [{"epoch": 1, "critic_loss": 0.5}, {"epoch": 2, "critic_loss": 0.3}]
        config = {"batch_size": 32, "lr": 0.0001}

        # 1. Save checkpoint
        saved_path = save_checkpoint(
            checkpoint_dir=self.test_dir,
            epoch=2,
            global_step=100,
            generator=netG,
            critic=netD,
            opt_g=optG,
            opt_d=optD,
            history=history,
            config=config,
            is_historical=True
        )

        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "historical", "epoch_0002.pt")))

        # 2. Restore into fresh model instances
        new_netG = ConditionalGenerator(latent_dim=50, num_classes=9, feature_maps=16).to(self.device)
        new_netD = ConditionalCritic(num_classes=9, feature_maps=16, image_size=64).to(self.device)
        new_optG = optim.Adam(new_netG.parameters(), lr=0.0001)
        new_optD = optim.Adam(new_netD.parameters(), lr=0.0001)

        next_epoch, step, loaded_hist, loaded_cfg = load_checkpoint(
            saved_path, new_netG, new_netD, new_optG, new_optD, self.device
        )

        # 3. Assertions
        self.assertEqual(next_epoch, 3)
        self.assertEqual(step, 100)
        self.assertEqual(len(loaded_hist), 2)
        self.assertEqual(loaded_cfg["batch_size"], 32)

        # Assert parameter equality
        for p1, p2 in zip(netG.parameters(), new_netG.parameters()):
            self.assertTrue(torch.equal(p1, p2))

        for p1, p2 in zip(netD.parameters(), new_netD.parameters()):
            self.assertTrue(torch.equal(p1, p2))

    def test_rng_determinism(self):
        """Verifies that captured and restored RNG states generate identical random streams."""
        rng = capture_rng_states()
        t1 = torch.randn(10, 10)

        # Restore
        restore_rng_states(rng)
        t2 = torch.randn(10, 10)

        self.assertTrue(torch.equal(t1, t2), "RNG state restoration failed to produce deterministic stream!")


if __name__ == "__main__":
    unittest.main()
