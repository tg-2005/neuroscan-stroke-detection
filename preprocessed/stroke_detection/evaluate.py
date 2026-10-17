"""Evaluate trained checkpoints and export diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .dataset import build_dataloaders, build_datasets
from .engine import evaluate_epoch
from .gradcam import generate_gradcam, save_gradcam_figure
from .metrics import collect_failure_cases, save_classification_report, save_confusion_matrix
from .models import load_model_from_checkpoint, normalize_model_name
from .preprocessing import PreprocessConfig
from .utils import get_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate stroke CT classifiers.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/preprocessed"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", choices=["googlenet", "inception_v3"], required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("D:/preprocessed/outputs/evaluation"))
    parser.add_argument("--gradcam-failures", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    model_name = normalize_model_name(args.model)
    model, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    datasets = build_datasets(args.data_root, preprocess_config=PreprocessConfig(), validate_images=True)
    loaders, class_weights, _ = build_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=False,
        ischemia_loss_boost=1.0,
    )
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    summary, metrics, y_true, y_pred, y_prob, paths = evaluate_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        desc="test",
    )

    output_dir = args.output_dir / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json({"checkpoint": str(args.checkpoint), "summary": summary, "metrics": metrics}, output_dir / "test_metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], output_dir / "confusion_matrix.png")
    save_classification_report(metrics, output_dir / "classification_report.csv")

    failures = collect_failure_cases(y_true, y_pred, y_prob, paths)
    failures.to_csv(output_dir / "failure_analysis.csv", index=False)

    for idx, row in failures.head(args.gradcam_failures).iterrows():
        predicted_idx = int(np.argmax([row["bleeding_prob"], row["ischemia_prob"], row["normal_prob"]]))
        result = generate_gradcam(
            model,
            model_name,
            Path(row["path"]),
            device,
            class_index=predicted_idx,
            preprocess_config=PreprocessConfig(),
        )
        safe_name = f"{idx:03d}_{row['actual']}_as_{row['predicted']}.png".replace(" ", "_")
        save_gradcam_figure(result, output_dir / "failure_gradcams" / safe_name)

    print(f"Evaluation complete. Accuracy={summary['accuracy']:.4f}, "
          f"Ischemia recall={summary['ischemia_recall']:.4f}, "
          f"Stroke sensitivity={summary['stroke_sensitivity']:.4f}")
    print(f"Diagnostics saved to {output_dir}")


if __name__ == "__main__":
    main()
