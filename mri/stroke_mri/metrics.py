from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_specificity(y_true: np.ndarray, y_pred: np.ndarray, normal_index: int = 0) -> float:
    negative = y_true == normal_index
    if negative.sum() == 0:
        return float("nan")
    return float((y_pred[negative] == normal_index).sum() / negative.sum())


def classwise_sensitivity_specificity(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        positive = y_true == idx
        negative = ~positive
        tp = np.logical_and(positive, y_pred == idx).sum()
        fn = np.logical_and(positive, y_pred != idx).sum()
        tn = np.logical_and(negative, y_pred != idx).sum()
        fp = np.logical_and(negative, y_pred == idx).sum()
        out[name] = {
            "sensitivity": float(tp / max(tp + fn, 1)),
            "specificity": float(tn / max(tn + fp, 1)),
            "false_negatives": int(fn),
            "false_positives": int(fp),
        }
    return out


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    average = "binary" if len(class_names) == 2 else "macro"
    positive_label = class_names.index("Stroke") if "Stroke" in class_names else 1
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, pos_label=positive_label, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, average=average, pos_label=positive_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, pos_label=positive_label, zero_division=0)),
        "specificity_normal": binary_specificity(y_true, y_pred, normal_index=class_names.index("Normal") if "Normal" in class_names else 0),
        "classwise_sensitivity_specificity": classwise_sensitivity_specificity(y_true, y_pred, class_names),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True),
    }
    try:
        if len(class_names) == 2:
            stroke_index = 1 if class_names[0] == "Normal" else 0
            metrics["roc_auc"] = float(roc_auc_score(y_true == stroke_index, y_score[:, stroke_index]))
        else:
            metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def plot_learning_curves(history: list[dict[str, float]], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(df["epoch"], df["train_acc"], label="train")
    axes[1].plot(df["epoch"], df["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_confusion_matrix(cm: list[list[int]], class_names: list[str], out_path: str | Path) -> None:
    cm_arr = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(cm_arr.shape[0]):
        for j in range(cm_arr.shape[1]):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
