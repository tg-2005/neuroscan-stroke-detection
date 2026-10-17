"""Evaluate the GoogleNet + InceptionV3 soft-voting ensemble on the test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CLASS_NAMES
from .dataset import build_datasets
from .decision import DecisionConfig, apply_decision_config
from .metrics import collect_failure_cases, compute_metrics, save_classification_report, save_confusion_matrix
from .models import extract_logits, load_model_from_checkpoint
from .preprocessing import PreprocessConfig
from .utils import get_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate soft-voting ensemble.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/preprocessed"))
    parser.add_argument("--googlenet-checkpoint", type=Path, default=Path("D:/final/preprocessed/outputs/googlenet/best_model.pt"))
    parser.add_argument("--inception-checkpoint", type=Path, default=Path("D:/final/preprocessed/outputs/inception_v3/best_model.pt"))
    parser.add_argument("--hybrid-checkpoint", type=Path, default=Path("D:/final/preprocessed/outputs/hybrid_cbam_swin/best_model.pt"))
    parser.add_argument("--googlenet-weight", type=float, default=0.3)
    parser.add_argument("--inception-weight", type=float, default=0.35)
    parser.add_argument("--hybrid-weight", type=float, default=0.35)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--decision-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("D:/final/preprocessed/outputs/evaluation/ensemble_3model"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    datasets = build_datasets(args.data_root, preprocess_config=PreprocessConfig(), validate_images=True)

    active_models = {}
    raw_weights = {}
    if args.googlenet_checkpoint.exists():
        active_models["googlenet"], _ = load_model_from_checkpoint(args.googlenet_checkpoint, device)
        raw_weights["googlenet"] = args.googlenet_weight
    if args.inception_checkpoint.exists():
        active_models["inception_v3"], _ = load_model_from_checkpoint(args.inception_checkpoint, device)
        raw_weights["inception_v3"] = args.inception_weight
    if args.hybrid_checkpoint and args.hybrid_checkpoint.exists():
        active_models["hybrid_cbam_swin"], _ = load_model_from_checkpoint(args.hybrid_checkpoint, device)
        raw_weights["hybrid_cbam_swin"] = args.hybrid_weight
    if not active_models:
        raise FileNotFoundError("No valid model checkpoints found for ensemble evaluation.")

    weight_sum = sum(raw_weights.values())
    if weight_sum <= 0:
        raw_weights = {name: 1.0 for name in active_models}
        weight_sum = float(len(active_models))
    weights = {name: raw_weights[name] / weight_sum for name in active_models}

    loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true = []
    y_prob = []
    paths = []
    for batch in tqdm(loader, desc="ensemble-test"):
        images = batch["image"].to(device, non_blocking=True)
        batch_probs = torch.zeros((images.size(0), len(CLASS_NAMES)), device=device)
        with torch.no_grad():
            for model_name, model in active_models.items():
                model.eval()
                logits = extract_logits(model(images))
                batch_probs += weights[model_name] * torch.softmax(logits, dim=1)
        probs_np = batch_probs.cpu().numpy()
        y_true.extend(batch["label"].numpy().astype(int).tolist())
        y_prob.extend(probs_np)
        paths.extend(batch["path"])

    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    decision_config = DecisionConfig.from_json(args.decision_config) if args.decision_config else None
    y_prob_arr = apply_decision_config(y_prob_arr, decision_config)
    y_pred = np.argmax(y_prob_arr, axis=1).astype(int).tolist()
    metrics = compute_metrics(y_true, y_pred, y_prob_arr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "weights": {
                name: weights[name] for name in active_models
            },
            "checkpoints": {
                name: str(getattr(args, f"{name.replace('_v3', '').replace('_cbam_swin', '')}_checkpoint", getattr(args, f"{name}_checkpoint", "")))
                for name in active_models
            },
            "decision_config": str(args.decision_config) if args.decision_config else None,
            "metrics": metrics,
        },
        args.output_dir / "ensemble_test_metrics.json",
    )
    save_confusion_matrix(metrics["confusion_matrix"], args.output_dir / "ensemble_confusion_matrix.png")
    save_classification_report(metrics, args.output_dir / "ensemble_classification_report.csv")
    failures = collect_failure_cases(y_true, y_pred, y_prob_arr, paths)
    failures.to_csv(args.output_dir / "ensemble_failure_analysis.csv", index=False)
    print(
        f"Ensemble complete. Accuracy={metrics['accuracy']:.4f}, "
        f"Ischemia recall={metrics['ischemia_recall']:.4f}, "
        f"Stroke sensitivity={metrics['stroke_sensitivity']:.4f}"
    )
    print(f"Class order: {CLASS_NAMES}")


if __name__ == "__main__":
    main()
