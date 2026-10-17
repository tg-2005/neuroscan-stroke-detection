"""Evaluation metrics and diagnostic reporting for stroke CT classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from .constants import CLASS_NAMES, CLASS_TO_IDX, STROKE_CLASS_INDICES


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
) -> Dict[str, object]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(len(CLASS_NAMES))))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )

    specificity = []
    for class_idx in range(len(CLASS_NAMES)):
        tp = cm[class_idx, class_idx]
        fp = cm[:, class_idx].sum() - tp
        fn = cm[class_idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        specificity.append(float(tn / (tn + fp)) if (tn + fp) else 0.0)

    stroke_true = np.isin(y_true_arr, STROKE_CLASS_INDICES).astype(int)
    stroke_pred = np.isin(y_pred_arr, STROKE_CLASS_INDICES).astype(int)
    stroke_cm = confusion_matrix(stroke_true, stroke_pred, labels=[0, 1])
    stroke_tn, stroke_fp, stroke_fn, stroke_tp = stroke_cm.ravel()
    stroke_sensitivity = stroke_tp / max(stroke_tp + stroke_fn, 1)
    stroke_specificity = stroke_tn / max(stroke_tn + stroke_fp, 1)

    roc_auc: Dict[str, float] = {}
    try:
        y_bin = label_binarize(y_true_arr, classes=list(range(len(CLASS_NAMES))))
        roc_auc["macro_ovr"] = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        for idx, class_name in enumerate(CLASS_NAMES):
            if len(np.unique(y_bin[:, idx])) == 2:
                roc_auc[class_name] = float(roc_auc_score(y_bin[:, idx], y_prob[:, idx]))
    except ValueError:
        roc_auc["macro_ovr"] = float("nan")

    classwise = {
        CLASS_NAMES[idx]: {
            "precision": float(precision[idx]),
            "recall_sensitivity": float(recall[idx]),
            "specificity": float(specificity[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(CLASS_NAMES))
    }

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "ischemia_recall": classwise["Ischemia"]["recall_sensitivity"],
        "stroke_sensitivity": float(stroke_sensitivity),
        "stroke_specificity": float(stroke_specificity),
        "classwise": classwise,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true_arr,
            y_pred_arr,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
    }


def medical_selection_score(metrics: Dict[str, object]) -> float:
    """Prefer clinically useful sensitivity without accepting all-stroke collapse.

    A model that predicts every image as stroke can achieve excellent stroke
    sensitivity while being unusable in practice. This score keeps ischemia
    recall important, but it also requires Normal recall / stroke specificity
    and macro F1 to protect against that degenerate solution.
    """
    classwise = metrics["classwise"]
    ischemia_recall = float(metrics["ischemia_recall"])
    stroke_sensitivity = float(metrics["stroke_sensitivity"])
    stroke_specificity = float(metrics["stroke_specificity"])
    macro_f1 = float(metrics["macro_f1"])
    normal_recall = float(classwise["Normal"]["recall_sensitivity"])
    bleeding_recall = float(classwise["Bleeding"]["recall_sensitivity"])

    score = (
        0.30 * ischemia_recall
        + 0.20 * stroke_sensitivity
        + 0.20 * stroke_specificity
        + 0.15 * normal_recall
        + 0.10 * macro_f1
        + 0.05 * bleeding_recall
    )
    if normal_recall < 0.10 or stroke_specificity < 0.10:
        score *= 0.25
    return score


def save_confusion_matrix(cm: Sequence[Sequence[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), labels=CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_classification_report(metrics: Dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = metrics["classification_report"]
    pd.DataFrame(report).transpose().to_csv(output_path)


def collect_failure_cases(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray,
    paths: Sequence[str],
    max_cases: int = 200,
    low_confidence_threshold: float = 0.60,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for true_idx, pred_idx, probs, path in zip(y_true, y_pred, y_prob, paths):
        confidence = float(np.max(probs))
        is_false_negative_stroke = true_idx in STROKE_CLASS_INDICES and pred_idx == CLASS_TO_IDX["Normal"]
        is_wrong = true_idx != pred_idx
        is_low_confidence = confidence < low_confidence_threshold
        if is_wrong or is_low_confidence:
            rows.append(
                {
                    "path": path,
                    "actual": CLASS_NAMES[true_idx],
                    "predicted": CLASS_NAMES[pred_idx],
                    "confidence": confidence,
                    "bleeding_prob": float(probs[0]),
                    "ischemia_prob": float(probs[1]),
                    "normal_prob": float(probs[2]),
                    "false_negative_stroke": is_false_negative_stroke,
                    "low_confidence": is_low_confidence,
                }
            )
    rows.sort(key=lambda row: (not row["false_negative_stroke"], row["confidence"]))
    return pd.DataFrame(rows[:max_cases])


def softmax_numpy(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits.detach().cpu(), dim=1).numpy()
