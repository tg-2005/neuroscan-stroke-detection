"""Decision threshold tuning for clinically safer inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .constants import CLASS_NAMES, CLASS_TO_IDX
from .metrics import compute_metrics
from .utils import save_json


@dataclass(frozen=True)
class DecisionConfig:
    class_multipliers: tuple[float, float, float] = (1.0, 1.0, 1.0)
    min_stroke_probability: Optional[float] = None

    @classmethod
    def from_json(cls, path: Path) -> "DecisionConfig":
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            class_multipliers=tuple(float(x) for x in payload["class_multipliers"]),
            min_stroke_probability=(
                None if payload.get("min_stroke_probability") is None else float(payload["min_stroke_probability"])
            ),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "class_multipliers": list(self.class_multipliers),
            "min_stroke_probability": self.min_stroke_probability,
        }


def apply_decision_config(probabilities: np.ndarray, config: Optional[DecisionConfig]) -> np.ndarray:
    if config is None:
        return probabilities
    probs = np.asarray(probabilities, dtype=np.float64)
    adjusted = probs * np.asarray(config.class_multipliers, dtype=np.float64)
    adjusted = adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
    if config.min_stroke_probability is not None:
        stroke_prob = probs[:, CLASS_TO_IDX["Bleeding"]] + probs[:, CLASS_TO_IDX["Ischemia"]]
        normal_idx = CLASS_TO_IDX["Normal"]
        force_stroke = (np.argmax(adjusted, axis=1) == normal_idx) & (stroke_prob >= config.min_stroke_probability)
        if np.any(force_stroke):
            stroke_scores = adjusted[force_stroke][:, [CLASS_TO_IDX["Bleeding"], CLASS_TO_IDX["Ischemia"]]]
            forced = np.where(
                stroke_scores[:, 0] >= stroke_scores[:, 1],
                CLASS_TO_IDX["Bleeding"],
                CLASS_TO_IDX["Ischemia"],
            )
            adjusted[force_stroke, normal_idx] = 0.0
            for row_idx, class_idx in zip(np.where(force_stroke)[0], forced):
                adjusted[row_idx, class_idx] = max(adjusted[row_idx, class_idx], 1.0)
            adjusted = adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
    return adjusted.astype(np.float32)


def tune_decision_config(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_ischemia_recall: float = 0.85,
    min_stroke_sensitivity: float = 0.90,
) -> tuple[DecisionConfig, Dict[str, object]]:
    """Grid-search class multipliers on validation probabilities.

    This does not retrain the model; it moves the operating point toward safer
    ischemia/stroke sensitivity while keeping accuracy and specificity visible.
    """
    best_config = DecisionConfig()
    best_metrics = compute_metrics(y_true, np.argmax(y_prob, axis=1), y_prob)
    best_score = _score_candidate(best_metrics, min_ischemia_recall, min_stroke_sensitivity)

    bleeding_grid = np.linspace(0.85, 1.20, 8)
    ischemia_grid = np.linspace(1.00, 1.70, 11)
    normal_grid = np.linspace(0.70, 1.15, 8)
    stroke_threshold_grid = [None, 0.30, 0.35, 0.40, 0.45]

    for bleeding_mult in bleeding_grid:
        for ischemia_mult in ischemia_grid:
            for normal_mult in normal_grid:
                for threshold in stroke_threshold_grid:
                    config = DecisionConfig(
                        class_multipliers=(float(bleeding_mult), float(ischemia_mult), float(normal_mult)),
                        min_stroke_probability=threshold,
                    )
                    adjusted = apply_decision_config(y_prob, config)
                    metrics = compute_metrics(y_true, np.argmax(adjusted, axis=1), adjusted)
                    score = _score_candidate(metrics, min_ischemia_recall, min_stroke_sensitivity)
                    if score > best_score:
                        best_score = score
                        best_config = config
                        best_metrics = metrics
    return best_config, best_metrics


def _score_candidate(metrics: Dict[str, object], min_ischemia_recall: float, min_stroke_sensitivity: float) -> float:
    ischemia_recall = float(metrics["ischemia_recall"])
    stroke_sensitivity = float(metrics["stroke_sensitivity"])
    stroke_specificity = float(metrics["stroke_specificity"])
    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    penalty = 0.0
    if ischemia_recall < min_ischemia_recall:
        penalty += 4.0 * (min_ischemia_recall - ischemia_recall)
    if stroke_sensitivity < min_stroke_sensitivity:
        penalty += 3.0 * (min_stroke_sensitivity - stroke_sensitivity)
    return 0.35 * macro_f1 + 0.25 * accuracy + 0.25 * ischemia_recall + 0.15 * stroke_specificity - penalty


def save_decision_config(path: Path, config: DecisionConfig, metrics: Dict[str, object]) -> None:
    save_json(
        {
            **config.to_dict(),
            "validation_metrics_after_thresholding": metrics,
            "class_order": CLASS_NAMES,
        },
        path,
    )
