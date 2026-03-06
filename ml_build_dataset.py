#!/usr/bin/env python3
"""
Step 1 (ML): Build a subject-level training table by merging
batch-extracted features with ICF targets.

This script is intentionally model-agnostic and focuses only on data assembly.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_numeric_columns(df: pd.DataFrame) -> List[str]:
    return [
        col
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
        and col not in {"activity_idx", "resting_idx", "t_start", "t_end", "duration_sec"}
    ]


def _aggregate_metrics_file(metrics_path: Path, prefix: str) -> Dict[str, float]:
    if not metrics_path.exists():
        return {}

    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}

    numeric_cols = _safe_numeric_columns(df)
    if not numeric_cols:
        return {}

    features: Dict[str, float] = {}
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        features[f"{prefix}_{col}_mean"] = float(values.mean())
        features[f"{prefix}_{col}_std"] = float(values.std(ddof=0))
        features[f"{prefix}_{col}_median"] = float(values.median())

    features[f"{prefix}_n_windows"] = float(len(df))
    return features


def _collect_subject_features(subject_dir: Path) -> Dict[str, float]:
    features: Dict[str, float] = {}

    # Core activity metrics
    features.update(_aggregate_metrics_file(subject_dir / "propulsion_hr_metrics.csv", "propulsion"))
    features.update(_aggregate_metrics_file(subject_dir / "resting_hr_metrics.csv", "resting"))

    # Extra activities: activity_<name>_hr_metrics.csv
    for file in sorted(subject_dir.glob("activity_*_hr_metrics.csv")):
        stem = file.stem
        # stem format: activity_<name>_hr_metrics
        activity_name = stem.removeprefix("activity_").removesuffix("_hr_metrics")
        prefix = f"custom_{activity_name}"
        features.update(_aggregate_metrics_file(file, prefix))

    return features


def find_latest_batch(batch_root: Path) -> Path:
    if not batch_root.exists():
        raise FileNotFoundError(f"Batch output root not found: {batch_root}")

    candidates = [d for d in batch_root.iterdir() if d.is_dir() and d.name.startswith("batch_")]
    if not candidates:
        raise FileNotFoundError(f"No batch directories found in {batch_root}")

    return sorted(candidates, key=lambda p: p.name)[-1]


def load_icf_targets(icf_csv: Path, id_col: str | int, target_col: str) -> pd.DataFrame:
    icf_df = pd.read_csv(icf_csv)

    if isinstance(id_col, int):
        if id_col < 0 or id_col >= len(icf_df.columns):
            raise ValueError(f"id_col index {id_col} is out of range for ICF CSV columns")
        subject_col = icf_df.columns[id_col]
    else:
        subject_col = id_col

    if subject_col not in icf_df.columns:
        raise ValueError(f"Subject ID column '{subject_col}' not found in ICF CSV")
    if target_col not in icf_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in ICF CSV")

    targets = icf_df[[subject_col, target_col]].copy()
    targets.rename(columns={subject_col: "subject_id", target_col: "target_score"}, inplace=True)
    targets["subject_id"] = (
        targets["subject_id"]
        .astype(str)
        .str.strip()
        .str.replace("^subj_", "sub_", regex=True)
        .str.replace("^Subject_", "sub_", regex=True)
    )
    targets["target_score"] = pd.to_numeric(targets["target_score"], errors="coerce")
    targets = targets.dropna(subset=["subject_id", "target_score"]).reset_index(drop=True)
    return targets


def build_training_table(batch_dir: Path, icf_targets: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    target_subjects = set(icf_targets["subject_id"].unique())

    for subject_dir in sorted(batch_dir.glob("sub_*")):
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name
        if subject_id not in target_subjects:
            continue

        feature_row = _collect_subject_features(subject_dir)
        if not feature_row:
            logger.warning("No features found for %s (skipping)", subject_id)
            continue

        feature_row["subject_id"] = subject_id
        rows.append(feature_row)

    features_df = pd.DataFrame(rows)
    if features_df.empty:
        return pd.DataFrame(columns=["subject_id", "target_score"])

    merged = features_df.merge(icf_targets, on="subject_id", how="inner")

    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    if "target_score" in numeric_cols:
        numeric_cols.remove("target_score")

    if numeric_cols:
        medians = merged[numeric_cols].median()
        merged[numeric_cols] = merged[numeric_cols].fillna(medians)

    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build subject-level ML training dataset")
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("./output_batch"),
        help="Root directory containing batch_* folders",
    )
    parser.add_argument(
        "--batch-dir",
        type=Path,
        default=None,
        help="Specific batch directory; if omitted, latest batch_* is used",
    )
    parser.add_argument(
        "--icf-csv",
        type=Path,
        default=Path(r"C:\Users\Nicla\Documents\ETHZ\Lifelogging\Data\ICF_scores_nursing_home.csv"),
        help="Path to ICF target CSV",
    )
    parser.add_argument(
        "--id-col",
        default="0",
        help="ICF subject ID column name or zero-based index (default: 0)",
    )
    parser.add_argument(
        "--target-col",
        required=True,
        help="ICF score column to use as first training target",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./output_ml/training_table.csv"),
        help="Output CSV path",
    )
    return parser.parse_args()


def _parse_id_col(value: str) -> str | int:
    value = value.strip()
    if value.isdigit():
        return int(value)
    return value


def main() -> None:
    args = parse_args()

    batch_dir = args.batch_dir if args.batch_dir is not None else find_latest_batch(args.batch_root)
    id_col = _parse_id_col(str(args.id_col))

    logger.info("Using batch directory: %s", batch_dir)
    logger.info("Using ICF CSV: %s", args.icf_csv)
    logger.info("Target column: %s", args.target_col)

    icf_targets = load_icf_targets(args.icf_csv, id_col=id_col, target_col=args.target_col)
    train_df = build_training_table(batch_dir, icf_targets)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(args.out, index=False)

    logger.info("Saved training table: %s", args.out)
    logger.info("Subjects in training table: %d", len(train_df))
    logger.info("Feature columns: %d", max(len(train_df.columns) - 2, 0))


if __name__ == "__main__":
    main()
