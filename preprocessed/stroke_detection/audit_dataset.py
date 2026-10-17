"""Audit dataset integrity before training."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import build_datasets
from .preprocessing import PreprocessConfig
from .utils import check_patient_leakage, class_counts, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CT stroke dataset splits.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/preprocessed"))
    parser.add_argument("--output", type=Path, default=Path("D:/preprocessed/outputs/dataset_audit.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = build_datasets(args.data_root, preprocess_config=PreprocessConfig(), validate_images=True)
    split_paths = {split: [path for path, _ in dataset.samples] for split, dataset in datasets.items()}
    report = {
        "splits": {
            split: {
                "root": str(dataset.split_dir),
                "num_images": len(dataset),
                "class_counts": class_counts(dataset.samples),
            }
            for split, dataset in datasets.items()
        },
        "patient_overlap": check_patient_leakage(split_paths),
    }
    save_json(report, args.output)
    print(report)
    print(f"Audit saved to {args.output}")


if __name__ == "__main__":
    main()
