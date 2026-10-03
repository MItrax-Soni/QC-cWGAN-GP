"""
Adaptive Quality-Controlled Filtering Orchestrator.
Executes the complete 4-stage filtering pipeline for minority facial expression classes:
Stage 1: Critic-Score Filtering
Stage 2: Feature-Space Similarity Filtering
Stage 3: Perceptual Hash Duplicate Detection
Stage 4: Feature-Space Diversity Maximization
"""

import os
import sys
import pandas as pd
from typing import Dict, List, Any, Tuple
import torch
import torchvision.utils as vutils

# Ensure root path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.yolo_parser import id_to_class_name
from src.data.preprocessing import denormalize_tensor
from src.filtering.critic_filter import CriticFilter
from src.filtering.similarity_filter import SimilarityFilter
from src.filtering.duplicate_filter import DuplicateFilter
from src.filtering.diversity_filter import DiversityFilter


class AdaptiveQualityFilter:
    """
    Coordinates multi-stage adaptive quality filtering and dataset augmentation.
    """
    def __init__(
        self,
        generator: torch.nn.Module,
        critic: torch.nn.Module,
        feature_extractor: torch.nn.Module,
        latent_dim: int = 100,
        device: torch.device = torch.device("cpu")
    ):
        self.generator = generator.to(device)
        self.critic = critic.to(device)
        self.feature_extractor = feature_extractor.to(device)
        self.latent_dim = latent_dim
        self.device = device

        self.stage1 = CriticFilter(self.critic, quantile=0.25, device=device)
        self.stage2 = SimilarityFilter(self.feature_extractor, min_cosine=0.20, max_cosine=0.95, device=device)
        self.stage3 = DuplicateFilter(max_hamming_distance=4)
        self.stage4 = DiversityFilter(self.feature_extractor, device=device)

    def calibrate(self, val_loader: torch.utils.data.DataLoader, train_loader: torch.utils.data.DataLoader) -> None:
        """Calibrates validation thresholds and builds real reference pools."""
        print("Calibrating Stage 1 validation thresholds...")
        self.stage1.calibrate_thresholds(val_loader)

        print("Building Stage 2 reference feature pool...")
        self.stage2.build_reference_pool(train_loader, max_per_class=200)

    def filter_class(
        self,
        class_id: int,
        num_candidates: int,
        target_accepted: int
    ) -> Dict[str, Any]:
        """
        Executes the 4 filtering stages for a single emotion class.
        """
        cname = id_to_class_name(class_id)
        print(f"\n[Filtering Pipeline] Processing Class {class_id} ({cname}) | Candidates: {num_candidates}...")

        # 1. Synthesize candidate pool
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(num_candidates, self.latent_dim, device=self.device)
            labels = torch.full((num_candidates,), class_id, dtype=torch.long, device=self.device)
            candidates = self.generator(z, labels).cpu()

        # 2. Stage 1: Critic-Score Filtering
        s1_acc, s1_rej, scores, thresh = self.stage1.filter_candidates(candidates, class_id)
        cand_s1 = candidates[s1_acc] if len(s1_acc) > 0 else torch.empty(0, 3, 64, 64)
        print(f" -> Stage 1 (Critic Filter, thresh={thresh:+.3f}): {len(cand_s1)} / {num_candidates} passed.")

        if len(cand_s1) == 0:
            return {
                "class_id": class_id, "class_name": cname, "candidates": num_candidates,
                "critic_accepted": 0, "similarity_accepted": 0, "unique_accepted": 0,
                "final_accepted": 0, "accepted_images": cand_s1, "rejected_images": candidates
            }

        # 3. Stage 2: Feature-Space Similarity Filtering
        s2_acc, s2_rej, sims = self.stage2.filter_candidates(cand_s1, class_id)
        cand_s2 = cand_s1[s2_acc] if len(s2_acc) > 0 else torch.empty(0, 3, 64, 64)
        print(f" -> Stage 2 (Similarity Filter [0.25, 0.95]): {len(cand_s2)} / {len(cand_s1)} passed.")

        if len(cand_s2) == 0:
            return {
                "class_id": class_id, "class_name": cname, "candidates": num_candidates,
                "critic_accepted": len(cand_s1), "similarity_accepted": 0, "unique_accepted": 0,
                "final_accepted": 0, "accepted_images": cand_s2, "rejected_images": candidates
            }

        # 4. Stage 3: Perceptual Hash Duplicate Filtering
        s3_acc, s3_rej = self.stage3.filter_candidates(cand_s2)
        cand_s3 = cand_s2[s3_acc] if len(s3_acc) > 0 else torch.empty(0, 3, 64, 64)
        print(f" -> Stage 3 (pHash Unique Filter): {len(cand_s3)} / {len(cand_s2)} passed.")

        if len(cand_s3) == 0:
            return {
                "class_id": class_id, "class_name": cname, "candidates": num_candidates,
                "critic_accepted": len(cand_s1), "similarity_accepted": len(cand_s2),
                "unique_accepted": 0, "final_accepted": 0, "accepted_images": cand_s3,
                "rejected_images": candidates
            }

        # 5. Stage 4: Feature-Space Diversity Maximization
        s4_acc, s4_rej = self.stage4.select_diverse_subset(cand_s3, target_accepted)
        final_accepted = cand_s3[s4_acc]
        print(f" -> Stage 4 (Diversity Maximization): {len(final_accepted)} final accepted.")

        return {
            "class_id": class_id,
            "class_name": cname,
            "candidates": num_candidates,
            "critic_accepted": len(cand_s1),
            "similarity_accepted": len(cand_s2),
            "unique_accepted": len(cand_s3),
            "final_accepted": len(final_accepted),
            "accepted_images": final_accepted,
            "rejected_images": candidates[s1_rej]
        }

    def execute_quality_filtering(
        self,
        minority_classes: List[int],
        candidates_per_class: int = 100,
        target_per_class: int = 40,
        output_dir: str = r"d:\Projects\GAN\data\synthetic"
    ) -> pd.DataFrame:
        """
        Executes adaptive filtering across all specified minority classes and saves accepted images.
        """
        os.makedirs(output_dir, exist_ok=True)
        stats = []
        manifest_rows = []

        for cid in minority_classes:
            res = self.filter_class(cid, candidates_per_class, target_per_class)
            cname = res["class_name"]

            stats.append({
                "Class": cname,
                "Class_ID": cid,
                "Candidates": res["candidates"],
                "Critic_Accepted": res["critic_accepted"],
                "Similarity_Accepted": res["similarity_accepted"],
                "Unique_Samples": res["unique_accepted"],
                "Final_Accepted": res["final_accepted"]
            })

            # Save accepted synthetic images
            cls_out = os.path.join(output_dir, cname)
            os.makedirs(cls_out, exist_ok=True)
            for idx in range(len(res["accepted_images"])):
                img_path = os.path.join(cls_out, f"syn_{cname}_{idx:04d}.png")
                denorm = denormalize_tensor(res["accepted_images"][idx])
                vutils.save_image(denorm, img_path)
                manifest_rows.append({
                    "image_path": img_path,
                    "class_id": cid,
                    "class_name": cname,
                    "is_synthetic": True
                })

        stats_df = pd.DataFrame(stats)
        table_path = r"d:\Projects\GAN\results\tables\filtering_statistics.csv"
        os.makedirs(os.path.dirname(table_path), exist_ok=True)
        stats_df.to_csv(table_path, index=False)
        print(f"\nFiltering statistics saved to: {table_path}")

        # Save synthetic manifest
        syn_manifest_df = pd.DataFrame(manifest_rows)
        syn_manifest_path = os.path.join(output_dir, "synthetic_manifest.csv")
        syn_manifest_df.to_csv(syn_manifest_path, index=False)
        print(f"Synthetic manifest saved: {len(syn_manifest_df)} samples -> {syn_manifest_path}")

        return stats_df


def run_adaptive_filtering_pipeline(
    checkpoint_path: str = r"d:\Projects\GAN\experiments\cwgan_gp_1000ep\checkpoints\latest.pt",
    val_manifest: str = r"d:\Projects\GAN\data\splits\valid_manifest.csv",
    train_manifest: str = r"d:\Projects\GAN\data\splits\pilot_manifest.csv",
    output_dir: str = r"d:\Projects\GAN\data\synthetic",
    minority_classes: List[int] = None,
    candidates_per_class: int = 50,
    target_per_class: int = 20,
    device: torch.device = None
):
    """Convenience top-level function executing the full adaptive filtering pipeline."""
    from src.models.generator import ConditionalGenerator
    from src.models.critic import ConditionalCritic
    from src.models.resnet_classifier import ResNetClassifier
    from src.data.dataset import get_dataloader
    from src.utils.device import get_device

    if device is None:
        device = get_device()
    if minority_classes is None:
        minority_classes = [1, 7]

    print(f"\n=======================================================")
    print(f"Executing 4-Stage Adaptive Quality Filtering Pipeline")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Minorities: {minority_classes}")
    print(f"=======================================================\n")

    gen = ConditionalGenerator(latent_dim=100, num_classes=9).to(device)
    critic = ConditionalCritic(num_classes=9).to(device)
    feat_extractor = ResNetClassifier(num_classes=9, pretrained=True).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    g_key = "generator_state" if "generator_state" in ckpt else "generator_state_dict"
    c_key = "critic_state" if "critic_state" in ckpt else "critic_state_dict"
    gen.load_state_dict(ckpt[g_key])
    critic.load_state_dict(ckpt[c_key])

    filter_system = AdaptiveQualityFilter(
        generator=gen,
        critic=critic,
        feature_extractor=feat_extractor,
        latent_dim=100,
        device=device
    )

    val_loader = get_dataloader(val_manifest, batch_size=32, shuffle=False)
    train_loader = get_dataloader(train_manifest, batch_size=32, shuffle=False)

    filter_system.calibrate(val_loader, train_loader)
    return filter_system.execute_quality_filtering(
        minority_classes=minority_classes,
        candidates_per_class=candidates_per_class,
        target_per_class=target_per_class,
        output_dir=output_dir
    )

