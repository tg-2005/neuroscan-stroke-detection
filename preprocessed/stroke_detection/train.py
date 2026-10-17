"""Train GoogleNet, InceptionV3, or both for stroke CT classification."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import TrainConfig
from .dataset import build_dataloaders, build_datasets
from .engine import fit_model
from .models import build_model, normalize_model_name
from .preprocessing import PreprocessConfig
from .utils import get_device, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stroke CT classifiers.")
    parser.add_argument("--data-root", type=Path, default=Path("D:/preprocessed"))
    parser.add_argument("--output-dir", type=Path, default=Path("D:/preprocessed/outputs"))
    parser.add_argument("--model", choices=["googlenet", "inception_v3", "hybrid_cbam_swin", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--full-finetune-epoch", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--use-weighted-sampler", action="store_true")
    parser.add_argument("--disable-class-weights", action="store_true")
    parser.add_argument("--ischemia-loss-boost", type=float, default=1.10)
    parser.add_argument("--loss", choices=["cross_entropy", "focal"], default="focal")
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--enable-light-denoise", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    config = TrainConfig(
        data_root=args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        full_finetune_epoch=args.full_finetune_epoch,
        learning_rate=args.lr,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        pretrained=not args.no_pretrained,
        use_weighted_sampler=args.use_weighted_sampler,
        use_class_weights=not args.disable_class_weights,
        ischemia_loss_boost=args.ischemia_loss_boost,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    preprocess_config = PreprocessConfig(enable_very_light_denoising=args.enable_light_denoise)
    datasets = build_datasets(args.data_root, preprocess_config=preprocess_config, validate_images=True)
    loaders, class_weights, counts = build_dataloaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=config.use_weighted_sampler,
        ischemia_loss_boost=config.ischemia_loss_boost,
    )
    save_json({"class_counts": counts, "class_weights": class_weights.tolist()}, args.output_dir / "dataset_summary.json")
    print(f"Class counts: {counts}")
    print(f"Class weights: {class_weights.tolist()}")

    model_names = ["googlenet", "inception_v3"] if args.model == "both" else [normalize_model_name(args.model)]
    for model_name in model_names:
        model_config = TrainConfig(**{**config.to_dict(), "model_name": model_name})
        model_config.data_root = args.data_root
        model_config.output_dir = args.output_dir
        model = build_model(
            model_name,
            pretrained=config.pretrained,
            dropout=config.dropout,
            freeze_backbone=True,
        ).to(device)
        result = fit_model(model, model_name, loaders, class_weights, model_config, device)
        save_json(result, args.output_dir / model_name / "training_result.json")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
