---
trigger: always_on
---

# PROJECT RULES: ADAPTIVE QUALITY-CONTROLLED cWGAN-GP

You are working on a research-level PyTorch project titled "GAN-Based Synthetic Data Augmentation for Improving Model Generalization".

The proposed methodology is Adaptive Quality-Controlled Conditional WGAN-GP for Long-Tailed Facial Expression Recognition.

## 1. Development Principles
- Prioritize correctness, modularity, reproducibility, and computational efficiency.
- Follow the uploaded project report as the primary research specification.
- Use PyTorch for all deep learning implementations.
- Never generate the entire project in one uncontrolled step.
- Implement, test, debug, and verify each module before proceeding.

## 2. Dataset
- Dataset: 9 Facial Expressions for YOLO.
- Approximate size: 4 GB.
- Verify actual dataset structure, annotations, class mapping, and distribution before preprocessing.
- Never assume YOLO labels directly represent image-level labels.
- Preserve the original long-tailed training distribution.
- Prevent train-validation-test leakage.
- Use 64x64 RGB images normalized to [-1,1] for GAN training.

## 3. Research Methodology
Implement:
- VAE.
- Vanilla Conditional GAN.
- Conditional WGAN-GP.
- Adaptive quality filtering.
- ResNet-18 classifier.
- Baseline comparisons.
- Ablation studies.

The proposed cWGAN-GP must include:
- Conditional generator and critic.
- Noise dimension 100.
- Five critic updates per generator update.
- Adam optimizer.
- Learning rate 0.0001.
- Betas (0.0, 0.9).
- Gradient penalty coefficient 10.

Preserve report specifications as defaults. Make hardware-dependent settings configurable.

## 4. Extended Training
The target is approximately 1,000 epochs.

Training must support:
- Automatic checkpointing.
- Resume from the latest checkpoint.
- Optimizer and random-state restoration.
- Mixed precision where supported.
- GPU and RAM monitoring.
- NaN/Inf detection.
- Graceful interruption.
- Out-of-memory recovery.
- Persistent experiment logs.

Never restart a completed experiment from epoch 1 unnecessarily.

Never silently change hyperparameters or batch size.

## 5. Quality Filtering
Implement all four stages:
1. Critic-score filtering.
2. Feature-space similarity filtering.
3. Duplicate and perceptual hash detection.
4. Diversity maximization.

Use validation data for threshold selection. Never use test data for filtering or tuning.

Do not interpret critic scores as calibrated probabilities.

## 6. Research Integrity
- Never fabricate metrics, results, graphs, or convergence claims.
- Clearly distinguish actual results from assumptions.
- Never claim that 1,000 epochs guarantees convergence.
- Maintain reproducible configurations.
- Document all methodological deviations from the report.

## 7. Development and Verification
- Inspect the environment before training.
- Verify GPU availability and VRAM.
- Test dataset parsing.
- Validate model dimensions.
- Test gradient penalty.
- Test checkpoint recovery.
- Run a 5–10 epoch pilot before extended training.
- Verify actual execution before marking any task complete.

## 8. Persistent Project Tracking
Maintain IMPLEMENTATION_STATUS.md.

After each development phase, update:
- Completed modules.
- Files created or modified.
- Tests executed and their actual results.
- Current experiment status.
- Unresolved issues.
- Exact next steps.

Always inspect existing project files before modifying them.

Do not overwrite working implementations without justification.

Your objective is a reliable, reproducible research implementation, not merely code generation.