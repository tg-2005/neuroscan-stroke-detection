from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from stroke_mri.dataset import (
    MRIDataset,
    _canonical_label,
    collate_skip_bad,
    discover_images,
    is_generated_augmentation,
    leakage_report,
    make_weighted_sampler,
    resolve_class_spec,
)
from stroke_mri.hybrid_model import (
    build_hybrid_model,
    freeze_hybrid_backbones,
    unfreeze_hybrid_backbones,
)
from stroke_mri.inference import logits_to_probabilities
from stroke_mri.losses import ClassWeightedFocalLoss, build_loss_function
from stroke_mri.metrics import compute_metrics, plot_learning_curves, save_confusion_matrix
from stroke_mri.model import build_model as build_resnet_model, freeze_backbone as freeze_resnet, unfreeze_deep_layers as unfreeze_resnet
from stroke_mri.preprocessing import PreprocessConfig
from stroke_mri.utils import ensure_dir, get_device, load_config, save_json, set_seed


def load_split_manifests(data_root: Path, manifests_dir: Path | None, class_mode: str) -> tuple[dict[str, pd.DataFrame], list[str]]:
    if manifests_dir is not None:
        splits = {split: pd.read_csv(manifests_dir / f"{split}.csv") for split in ("train", "val", "test")}
        label_source = pd.concat(splits.values(), ignore_index=True)
        label_column = "label" if "label" in label_source.columns else "source_label"
        spec = resolve_class_spec(label_source[label_column], class_mode)
        class_names = spec.class_names
        for split, df in splits.items():
            df = df.copy()
            if "label" not in df.columns:
                df["label"] = df["source_label"]
            df["label"] = df["label"].map(lambda x: _canonical_label(str(x), spec.mode))
            df["target"] = df["label"].map(spec.class_to_idx)
            if df["target"].isna().any():
                missing = sorted(df.loc[df["target"].isna(), "label"].astype(str).unique())
                raise ValueError(f"Manifest labels not present in resolved class mapping: {missing}")
            df["target"] = df["target"].astype(int)
            splits[split] = df
        return splits, class_names

    df, spec = discover_images(data_root, class_mode)
    if not df["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Use prepare_data.py first, or pass a dataset root containing train/val/test folders.")
    report = leakage_report(df)
    if report["patient_overlap"] or report["duplicate_hash_overlap"]:
        raise RuntimeError(f"Data leakage detected: {report}")
    return {split: df[df["split"] == split].reset_index(drop=True) for split in ("train", "val", "test")}, spec.class_names


def reject_generated_augmented_leakage(splits: dict[str, pd.DataFrame]) -> None:
    flagged: list[str] = []
    for split_name, df in splits.items():
        bad = df[df["path"].map(is_generated_augmentation)]
        if not bad.empty:
            flagged.extend(bad["path"].head(3).tolist())
    if flagged:
        raise ValueError(
            "Found generated augmented files in split manifests. "
            "Regenerate manifests with prepare_data.py default settings, or set allow_generated_augmented_files: true "
            f"only after verifying patient-level provenance. Examples: {flagged}"
        )


def make_loss(
    class_names: list[str],
    train_df: pd.DataFrame,
    device: torch.device,
    stroke_weight: float,
    loss_type: str = "focal",
    gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> nn.Module:
    counts = train_df["target"].value_counts().sort_index()
    weights = len(train_df) / (len(class_names) * counts.reindex(range(len(class_names)), fill_value=1))
    for idx, name in enumerate(class_names):
        if name.lower() in {"ischemic", "stroke", "stroke_unknown", "hemorrhagic"}:
            weights.iloc[idx] *= stroke_weight
    class_weights = torch.tensor(weights.to_numpy(dtype=np.float32), device=device)

    if loss_type.lower() == "focal":
        return ClassWeightedFocalLoss(alpha=class_weights, gamma=gamma, label_smoothing=label_smoothing)

    if len(class_names) == 2:
        positive_idx = class_names.index("Stroke") if "Stroke" in class_names else 1
        positives = int((train_df["target"] == positive_idx).sum())
        negatives = len(train_df) - positives
        pos_weight = torch.tensor([(negatives / max(positives, 1)) * stroke_weight], device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)


def predict_from_probs(probs: np.ndarray, class_names: list[str], decision_threshold: float | None = None) -> np.ndarray:
    if decision_threshold is not None and class_names == ["Normal", "Stroke"]:
        stroke_index = class_names.index("Stroke")
        normal_index = class_names.index("Normal")
        return np.where(probs[:, stroke_index] >= decision_threshold, stroke_index, normal_index)
    return probs.argmax(axis=1)


def tune_binary_threshold(y_true: list[int], y_score: np.ndarray, class_names: list[str], min_sensitivity: float = 0.95) -> float:
    if class_names != ["Normal", "Stroke"]:
        return 0.5
    stroke_index = class_names.index("Stroke")
    y_true_arr = np.asarray(y_true) == stroke_index
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.10, 0.90, 81):
        y_pred = y_score[:, stroke_index] >= threshold
        tp = float(np.logical_and(y_true_arr, y_pred).sum())
        fn = float(np.logical_and(y_true_arr, ~y_pred).sum())
        fp = float(np.logical_and(~y_true_arr, y_pred).sum())
        sensitivity = tp / max(tp + fn, 1.0)
        precision = tp / max(tp + fp, 1.0)
        beta2 = 4.0
        f2 = (1 + beta2) * precision * sensitivity / max(beta2 * precision + sensitivity, 1e-8)
        score = f2 + (0.05 if sensitivity >= min_sensitivity else 0.0)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def run_epoch(model, loader, criterion, device, optimizer=None, class_names=None, decision_threshold: float | None = None):
    training = optimizer is not None
    model.train(training)
    losses, sample_count, y_true, y_pred, y_score = [], 0, [], [], []
    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(device)
        targets = batch["target"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            if len(class_names) == 2 and not isinstance(criterion, ClassWeightedFocalLoss):
                loss = criterion(logits[:, 0] if logits.ndim > 1 else logits, targets.float())
            else:
                loss = criterion(logits, targets)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        probs = logits_to_probabilities(logits.detach(), class_names).cpu().numpy()
        preds = predict_from_probs(probs, class_names, decision_threshold)
        batch_size = images.size(0)
        sample_count += batch_size
        losses.append(float(loss.item()) * batch_size)
        y_true.extend(targets.cpu().numpy().tolist())
        y_pred.extend(preds.tolist())
        y_score.append(probs)
    if sample_count == 0:
        raise RuntimeError("No readable images were available for this epoch.")
    return {
        "loss": sum(losses) / sample_count,
        "acc": accuracy_score(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": np.concatenate(y_score, axis=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Stroke MRI Classifier (ResNet50 or Hybrid CNN-Transformer).")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--manifests-dir", type=Path, default=None)
    parser.add_argument("--model-arch", type=str, default=None, choices=["hybrid_cbam_swin", "resnet50"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_arch is not None:
        cfg["model_arch"] = args.model_arch
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    model_arch = cfg.get("model_arch", "hybrid_cbam_swin")
    loss_type = cfg.get("loss_type", "focal")
    focal_gamma = float(cfg.get("focal_gamma", 2.0))
    label_smoothing = float(cfg.get("label_smoothing", 0.05))

    set_seed(int(cfg["seed"]))
    run_dir = ensure_dir(cfg["outputs"]["run_dir"])
    device = get_device()

    print(f"[INFO] Initializing Stroke AI training pipeline...")
    print(f"[INFO] Model Architecture : {model_arch}")
    print(f"[INFO] Loss Function      : {loss_type} (gamma={focal_gamma})")
    print(f"[INFO] Hardware Device    : {device}")

    splits, class_names = load_split_manifests(args.data_root, args.manifests_dir, cfg["class_mode"])
    if not bool(cfg.get("allow_generated_augmented_files", False)):
        reject_generated_augmented_leakage(splits)

    preprocess_cfg = PreprocessConfig(image_size=cfg["image_size"], **cfg["preprocessing"])
    datasets = {
        "train": MRIDataset(splits["train"], preprocess_cfg, augment=True),
        "val": MRIDataset(splits["val"], preprocess_cfg, augment=False),
        "test": MRIDataset(splits["test"], preprocess_cfg, augment=False),
    }

    sampler = make_weighted_sampler(splits["train"]) if cfg["use_weighted_sampler"] else None
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg["batch_size"], sampler=sampler, shuffle=sampler is None, num_workers=cfg["num_workers"], collate_fn=collate_skip_bad),
        "val": DataLoader(datasets["val"], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate_skip_bad),
        "test": DataLoader(datasets["test"], batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], collate_fn=collate_skip_bad),
    }

    # Instantiate selected model architecture
    if model_arch == "hybrid_cbam_swin":
        model = build_hybrid_model(
            num_classes=len(class_names),
            pretrained=True,
            dropout=float(cfg["dropout"]),
            embed_dim=int(cfg.get("embed_dim", 256)),
        ).to(device)
        freeze_hybrid_backbones(model)
    else:
        model = build_resnet_model(len(class_names), dropout=cfg["dropout"], pretrained=True).to(device)
        freeze_resnet(model)

    criterion = make_loss(
        class_names=class_names,
        train_df=splits["train"],
        device=device,
        stroke_weight=float(cfg["stroke_sensitivity_weight"]),
        loss_type=loss_type,
        gamma=focal_gamma,
        label_smoothing=label_smoothing,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)

    best_val_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, int(cfg["epochs"]) + 1):
        if epoch == int(cfg["freeze_backbone_epochs"]) + 1:
            print("[INFO] Warm-up complete. Unfreezing deep backbone layers for fine-tuning...")
            if model_arch == "hybrid_cbam_swin":
                unfreeze_hybrid_backbones(model)
            else:
                unfreeze_resnet(model)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg["learning_rate"] * 0.3,
                weight_decay=cfg["weight_decay"],
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=3)

        print(f"Epoch {epoch}/{cfg['epochs']}")
        train_out = run_epoch(model, loaders["train"], criterion, device, optimizer=optimizer, class_names=class_names)
        val_out = run_epoch(model, loaders["val"], criterion, device, optimizer=None, class_names=class_names)
        scheduler.step(val_out["loss"])
        row = {
            "epoch": epoch,
            "train_loss": train_out["loss"],
            "val_loss": val_out["loss"],
            "train_acc": train_out["acc"],
            "val_acc": val_out["acc"],
        }
        history.append(row)
        print(f"  Train Loss: {train_out['loss']:.4f} | Train Acc: {train_out['acc']:.4f}")
        print(f"  Val Loss  : {val_out['loss']:.4f} | Val Acc  : {val_out['acc']:.4f}")

        if val_out["loss"] < best_val_loss:
            best_val_loss = val_out["loss"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_arch": model_arch,
                    "class_names": class_names,
                    "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
                    "num_classes": len(class_names),
                    "preprocess_config": preprocess_cfg.__dict__,
                    "dropout": cfg["dropout"],
                    "decision_threshold": cfg["decision_threshold"],
                    "config": cfg,
                    "epoch": epoch,
                },
                run_dir / "best_model.pt",
            )
            print(f"  [*] Saved new best model checkpoint to {run_dir / 'best_model.pt'}")
        else:
            stale_epochs += 1
            if stale_epochs >= int(cfg["patience"]):
                print("Early stopping triggered.")
                break

    plot_learning_curves(history, run_dir / "learning_curves.png")
    save_json(history, run_dir / "history.json")

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    val_eval = run_epoch(model, loaders["val"], criterion, device, optimizer=None, class_names=class_names)
    tuned_threshold = tune_binary_threshold(val_eval["y_true"], val_eval["y_score"], class_names)
    checkpoint["decision_threshold"] = tuned_threshold if class_names == ["Normal", "Stroke"] else cfg["decision_threshold"]
    torch.save(checkpoint, run_dir / "best_model.pt")
    test_out = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        class_names=class_names,
        decision_threshold=checkpoint["decision_threshold"] if class_names == ["Normal", "Stroke"] else None,
    )
    metrics = compute_metrics(test_out["y_true"], test_out["y_pred"], test_out["y_score"], class_names)
    save_json(metrics, run_dir / "test_metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], class_names, run_dir / "confusion_matrix.png")
    print("\n=== Final Test Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
