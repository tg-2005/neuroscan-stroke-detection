"""Tune ensemble decision thresholds on the validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .constants import CLASS_NAMES
from .dataset import build_datasets
from .decision import apply_decision_config, save_decision_config, tune_decision_config
from .metrics import compute_metrics
from .models import extract_logits, load_model_from_checkpoint
from .preprocessing import PreprocessConfig
from .utils import get_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune class thresholds on validation data.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/preprocessed"))
    parser.add_argument("--googlenet-checkpoint", type=Path, default=Path("D:/preprocessed/outputs/googlenet/best_model.pt"))
    parser.add_argument("--inception-checkpoint", type=Path, default=Path("D:/preprocessed/outputs/inception_v3/best_model.pt"))
    parser.add_argument("--googlenet-weight", type=float, default=0.4)
    parser.add_argument("--inception-weight", type=float, default=0.6)
    parser.add_argument("--min-ischemia-recall", type=float, default=0.85)
    parser.add_argument("--min-stroke-sensitivity", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("D:/preprocessed/outputs/evaluation/ensemble/decision_thresholds.json"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    datasets = build_datasets(args.data_root, preprocess_config=PreprocessConfig(), validate_images=True)
    models = {}
    raw_weights = {}
    if args.googlenet_checkpoint.exists():
        models["googlenet"], _ = load_model_from_checkpoint(args.googlenet_checkpoint, device)
        raw_weights["googlenet"] = args.googlenet_weight
    if args.inception_checkpoint.exists():
        models["inception_v3"], _ = load_model_from_checkpoint(args.inception_checkpoint, device)
        raw_weights["inception_v3"] = args.inception_weight
    if not models:
        raise FileNotFoundError("No checkpoints found for threshold tuning.")
    total_weight = sum(raw_weights.values())
    weights = {name: raw_weights[name] / total_weight for name in models}

    loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    y_true = []
    y_prob = []
    for batch in tqdm(loader, desc="val-probs"):
        images = batch["image"].to(device, non_blocking=True)
        probs = torch.zeros((images.size(0), len(CLASS_NAMES)), device=device)
        with torch.no_grad():
            for model_name, model in models.items():
                model.eval()
                probs += weights[model_name] * torch.softmax(extract_logits(model(images)), dim=1)
        y_true.extend(batch["label"].numpy().astype(int).tolist())
        y_prob.extend(probs.cpu().numpy())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_prob_arr = np.asarray(y_prob, dtype=np.float32)
    base_metrics = compute_metrics(y_true_arr, np.argmax(y_prob_arr, axis=1), y_prob_arr)
    config, tuned_metrics = tune_decision_config(
        y_true_arr,
        y_prob_arr,
        min_ischemia_recall=args.min_ischemia_recall,
        min_stroke_sensitivity=args.min_stroke_sensitivity,
    )
    tuned_probs = apply_decision_config(y_prob_arr, config)
    tuned_metrics = compute_metrics(y_true_arr, np.argmax(tuned_probs, axis=1), tuned_probs)
    save_decision_config(args.output, config, tuned_metrics)

    print("Base validation:")
    print(
        f"Accuracy={base_metrics['accuracy']:.4f}, "
        f"Ischemia recall={base_metrics['ischemia_recall']:.4f}, "
        f"Stroke sensitivity={base_metrics['stroke_sensitivity']:.4f}, "
        f"Stroke specificity={base_metrics['stroke_specificity']:.4f}"
    )
    print("Tuned validation:")
    print(
        f"Accuracy={tuned_metrics['accuracy']:.4f}, "
        f"Ischemia recall={tuned_metrics['ischemia_recall']:.4f}, "
        f"Stroke sensitivity={tuned_metrics['stroke_sensitivity']:.4f}, "
        f"Stroke specificity={tuned_metrics['stroke_specificity']:.4f}"
    )
    print(f"Saved decision config to {args.output}")


if __name__ == "__main__":
    main()
