#!/usr/bin/env python3
"""
Step 3 (ML): Training scaffold for hybrid ICF model.

This script provides a minimal end-to-end training loop that wires:
- Step-2 data module
- Step-3 hybrid model
- Theory-informed loss

Current C_a inputs are placeholder-based and can be replaced with your final mapping.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from ml_data_module import ICFHybridDataModule, split_feature_columns
from ml_hybrid_model import (
    CapacityConfig,
    HybridICFModel,
    TheoryInformedICFLoss,
    compute_functional_capacity,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _to_icf_class_labels(target_scores: torch.Tensor, n_classes: int = 5) -> torch.Tensor:
    """
    Converts continuous 0-100 target to ordinal bins [0..n_classes-1].
    """
    clipped = target_scores.clamp(0.0, 100.0)
    bin_size = 100.0 / n_classes
    labels = torch.floor(clipped / bin_size).long().view(-1)
    labels = torch.clamp(labels, 0, n_classes - 1)
    return labels


def run_train_step(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logger.info("Using device: %s", device)

    table = pd.read_csv(args.table)
    split = split_feature_columns(table, target_col=args.target_col, subject_col=args.subject_col, drop_ppg=True)

    data_module = ICFHybridDataModule(
        table=table,
        feature_split=split,
        target_col=args.target_col,
        subject_col=args.subject_col,
        train_fraction=args.train_frac,
        val_fraction=args.val_frac,
        test_fraction=args.test_frac,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    data_module.setup()

    subject_ids = table[args.subject_col].astype(str).values
    train_subjects = {subject_ids[i] for i in data_module.indices["train"]}
    val_subjects = {subject_ids[i] for i in data_module.indices["val"]}
    test_subjects = {subject_ids[i] for i in data_module.indices["test"]}

    overlap_train_val = train_subjects & val_subjects
    overlap_train_test = train_subjects & test_subjects
    overlap_val_test = val_subjects & test_subjects

    print("\n--- Split Leakage Report ---")
    print(f"split_mode: {args.split_mode}")
    print(f"train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"val subjects ({len(val_subjects)}): {sorted(val_subjects)}")
    print(f"test subjects ({len(test_subjects)}): {sorted(test_subjects)}")
    print(f"overlap train-val ({len(overlap_train_val)}): {sorted(overlap_train_val)}")
    print(f"overlap train-test ({len(overlap_train_test)}): {sorted(overlap_train_test)}")
    print(f"overlap val-test ({len(overlap_val_test)}): {sorted(overlap_val_test)}")
    print("----------------------------\n")

    model = HybridICFModel(
        hrv_input_dim=max(len(split.hrv_columns), 1),
        sensor_token_dim=data_module.max_transformer_dim,
        hrv_hidden_dim=args.hrv_hidden_dim,
        sensor_model_dim=args.sensor_model_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        num_heads=args.num_heads,
        num_sensor_layers=args.num_sensor_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = TheoryInformedICFLoss(alpha=args.alpha, beta=args.beta, margin=args.margin)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    capacity_cfg = CapacityConfig(
        omega_f=args.omega_f,
        omega_q=args.omega_q,
        tau=args.ca_tau,
        expected_frequency=args.ca_expected_frequency,
        expected_duration=args.ca_expected_duration,
    )

    train_loader = data_module.dataloader("train", batch_size=args.batch_size, shuffle=True)
    model.train()

    batch = next(iter(train_loader))
    x_hrv = batch["hrv"].to(device)
    x_tokens = batch["transformer_tokens"].to(device)
    y_target = batch["target_score"].to(device)

    outputs = model(x_hrv=x_hrv, transformer_tokens=x_tokens)
    pred_score = outputs["pred_score"]

    # Placeholder C_a input plumbing (can be replaced with true p_i/d_i mapping)
    # p_i: pseudo-probabilities from sigmoid-normalized HRV values
    # d_i: pseudo-durations from positive token magnitudes
    pseudo_prob = torch.sigmoid(x_hrv)
    pseudo_duration = x_hrv.abs() + 1e-3
    ca_value = compute_functional_capacity(
        probabilities=pseudo_prob,
        durations=pseudo_duration,
        config=capacity_cfg,
    )

    # Scale C_a to 0-100 for compatibility with targets and predictions
    ca_value = torch.clamp(ca_value * 100.0, 0.0, 100.0)

    icf_class_label = _to_icf_class_labels(y_target)

    total_loss, l_base, l_ca, l_ordinal = criterion(
        pred_score=pred_score,
        clinical_target=y_target,
        ca_value=ca_value,
        icf_class_label=icf_class_label,
    )

    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    logger.info("One-step training scaffold successful")
    logger.info(
        "Losses -> total: %.4f | base: %.4f | ca: %.4f | ordinal: %.4f",
        float(total_loss.item()),
        float(l_base.item()),
        float(l_ca.item()),
        float(l_ordinal.item()),
    )
    logger.info("Shapes -> HRV: %s | tokens: %s | pred: %s", tuple(x_hrv.shape), tuple(x_tokens.shape), tuple(pred_score.shape))
    logger.info("Feature counts -> HRV: %d, EDA: %d, IMU: %d", len(split.hrv_columns), len(split.eda_columns), len(split.imu_columns))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid ICF model training scaffold")

    parser.add_argument("--table", type=Path, default=Path("./output_ml/training_table.csv"), help="Training table CSV")
    parser.add_argument("--subject-col", default="subject_id", help="Subject ID column")
    parser.add_argument("--target-col", default="target_score", help="Target column")

    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["subject", "row"], default="subject")
    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--hrv-hidden-dim", type=int, default=128)
    parser.add_argument("--sensor-model-dim", type=int, default=128)
    parser.add_argument("--fusion-hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-sensor-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--alpha", type=float, default=0.5, help="Theory-informed C_a loss weight")
    parser.add_argument("--beta", type=float, default=0.3, help="Ordinal contrastive loss weight")
    parser.add_argument("--margin", type=float, default=15.0, help="Ordinal margin")

    parser.add_argument("--omega-f", type=float, default=0.5, help="C_a frequency term weight")
    parser.add_argument("--omega-q", type=float, default=0.5, help="C_a quality term weight")
    parser.add_argument("--ca-tau", type=float, default=0.5, help="C_a threshold")
    parser.add_argument("--ca-expected-frequency", type=float, default=1.0, help="C_a expected frequency")
    parser.add_argument("--ca-expected-duration", type=float, default=1.0, help="C_a expected duration")

    parser.add_argument("--force-cpu", action="store_true", help="Force CPU even if CUDA is available")

    return parser.parse_args()


if __name__ == "__main__":
    run_train_step(parse_args())
