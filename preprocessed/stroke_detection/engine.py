"""Training and validation engine."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import TrainConfig
from .metrics import compute_metrics, medical_selection_score, softmax_numpy
from .models import (
    compute_auxiliary_loss,
    extract_logits,
    save_checkpoint,
    unfreeze_all_layers,
    unfreeze_deep_layers,
)
from .losses import build_loss
from .utils import ensure_dir, save_json


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = compute_auxiliary_loss(outputs, labels, criterion)
        loss.backward()
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        logits = extract_logits(outputs)
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((torch.argmax(logits, dim=1) == labels).sum().item())
        total_seen += images.size(0)

    return {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> Tuple[Dict[str, float], Dict[str, object], List[int], List[int], np.ndarray, List[str]]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob_batches: List[np.ndarray] = []
    paths: List[str] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        logits = extract_logits(outputs)
        loss = criterion(logits, labels)
        probs = softmax_numpy(logits)
        preds = np.argmax(probs, axis=1)

        total_loss += float(loss.item()) * images.size(0)
        total_seen += images.size(0)
        y_true.extend(labels.cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.astype(int).tolist())
        y_prob_batches.append(probs)
        paths.extend(batch["path"])

    y_prob = np.concatenate(y_prob_batches, axis=0)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    summary = {
        "loss": total_loss / max(total_seen, 1),
        "accuracy": float(metrics["accuracy"]),
        "ischemia_recall": float(metrics["ischemia_recall"]),
        "stroke_sensitivity": float(metrics["stroke_sensitivity"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    return summary, metrics, y_true, y_pred, y_prob, paths


def fit_model(
    model: nn.Module,
    model_name: str,
    loaders: Dict[str, torch.utils.data.DataLoader],
    class_weights: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
) -> Dict[str, object]:
    output_dir = ensure_dir(config.output_dir / model_name)
    criterion_weights = class_weights.to(device) if config.use_class_weights else None
    criterion = build_loss(
        config.loss_name,
        criterion_weights,
        focal_gamma=config.focal_gamma,
        label_smoothing=config.label_smoothing,
    )
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=config.min_learning_rate,
    )

    history: List[Dict[str, float]] = []
    best_score = -float("inf")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    start = time.time()

    for epoch in range(1, config.num_epochs + 1):
        if epoch == config.freeze_epochs + 1:
            unfreeze_deep_layers(model, model_name)
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=config.learning_rate * 0.25,
                weight_decay=config.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=config.min_learning_rate,
            )

        if config.full_finetune_epoch > 0 and epoch == config.full_finetune_epoch:
            unfreeze_all_layers(model)
            optimizer = AdamW(
                model.parameters(),
                lr=config.learning_rate * 0.05,
                weight_decay=config.weight_decay,
            )
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                min_lr=config.min_learning_rate,
            )

        print(f"\nEpoch {epoch}/{config.num_epochs} [{model_name}]")
        train_summary = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            config.max_grad_norm,
        )
        val_summary, val_metrics, *_ = evaluate_epoch(model, loaders["val"], criterion, device, desc="val")
        scheduler.step(val_summary["loss"])

        row = {
            "epoch": float(epoch),
            "train_loss": train_summary["loss"],
            "train_accuracy": train_summary["accuracy"],
            "val_loss": val_summary["loss"],
            "val_accuracy": val_summary["accuracy"],
            "val_ischemia_recall": val_summary["ischemia_recall"],
            "val_stroke_sensitivity": val_summary["stroke_sensitivity"],
            "val_macro_f1": val_summary["macro_f1"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            "loss "
            f"{row['train_loss']:.4f}/{row['val_loss']:.4f} | "
            f"acc {row['train_accuracy']:.4f}/{row['val_accuracy']:.4f} | "
            f"ischemia recall {row['val_ischemia_recall']:.4f} | "
            f"stroke sensitivity {row['val_stroke_sensitivity']:.4f}"
        )

        score = medical_selection_score(val_metrics)
        improved = score > best_score or (np.isclose(score, best_score) and val_summary["loss"] < best_val_loss)
        if improved:
            best_score = score
            best_val_loss = val_summary["loss"]
            epochs_without_improvement = 0
            save_checkpoint(
                output_dir / "best_model.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                model_name,
                {"selection_score": best_score, **val_summary},
                config.to_dict(),
            )
            save_json(val_metrics, output_dir / "best_val_metrics.json")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_dir / "last_model.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            model_name,
            {"selection_score": score, **val_summary},
            config.to_dict(),
        )
        save_history_and_curves(history, output_dir)

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping after {config.patience} epochs without medical-score improvement.")
            break

    return {
        "history": history,
        "best_score": best_score,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": time.time() - start,
        "best_checkpoint": str(output_dir / "best_model.pt"),
    }


def save_history_and_curves(history: List[Dict[str, float]], output_dir: Path) -> None:
    import pandas as pd

    ensure_dir(output_dir)
    df = pd.DataFrame(history)
    df.to_csv(output_dir / "training_history.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(df["epoch"], df["train_accuracy"], label="train acc")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val acc")
    axes[1].plot(df["epoch"], df["val_ischemia_recall"], label="ischemia recall")
    axes[1].plot(df["epoch"], df["val_stroke_sensitivity"], label="stroke sensitivity")
    axes[1].set_title("Medical Performance")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)
