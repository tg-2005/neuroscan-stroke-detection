"""Single-model and ensemble inference for CT stroke detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import torch

from .config import EnsembleConfig
from .constants import CLASS_NAMES
from .decision import DecisionConfig, apply_decision_config
from .gradcam import GradCAMResult, generate_gradcam
from .models import extract_logits, load_model_from_checkpoint
from .preprocessing import PreprocessConfig, preprocess_ct_image
from .utils import get_device


@dataclass
class PredictionResult:
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: Dict[str, float]
    model_probabilities: Dict[str, Dict[str, float]]
    gradcams: Dict[str, GradCAMResult]


class StrokePredictor:
    def __init__(
        self,
        googlenet_checkpoint: Optional[Path] = None,
        inception_v3_checkpoint: Optional[Path] = None,
        hybrid_checkpoint: Optional[Path] = None,
        ensemble_weights: Sequence[float] = (0.3, 0.35, 0.35),
        device: str = "auto",
        preprocess_config: Optional[PreprocessConfig] = None,
        decision_config: Optional[DecisionConfig] = None,
        gradcam_resolution: str = "balanced",
    ) -> None:
        self.device = get_device(device)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.decision_config = decision_config
        self.gradcam_resolution = gradcam_resolution
        self.models: Dict[str, torch.nn.Module] = {}
        self.weights: Dict[str, float] = {}

        if googlenet_checkpoint is not None and Path(googlenet_checkpoint).exists():
            model, _ = load_model_from_checkpoint(Path(googlenet_checkpoint), self.device)
            self.models["googlenet"] = model
        if inception_v3_checkpoint is not None and Path(inception_v3_checkpoint).exists():
            model, _ = load_model_from_checkpoint(Path(inception_v3_checkpoint), self.device)
            self.models["inception_v3"] = model
        if hybrid_checkpoint is not None and Path(hybrid_checkpoint).exists():
            model, _ = load_model_from_checkpoint(Path(hybrid_checkpoint), self.device)
            self.models["hybrid_cbam_swin"] = model

        if not self.models:
            raise FileNotFoundError("No valid checkpoints were provided for inference.")

        names = ["googlenet", "inception_v3", "hybrid_cbam_swin"]
        raw_weights = {name: float(weight) for name, weight in zip(names, ensemble_weights)}
        active_sum = sum(raw_weights.get(name, 0.0) for name in self.models)
        if active_sum <= 0:
            active_sum = float(len(self.models))
            raw_weights = {name: 1.0 for name in self.models}
        self.weights = {name: raw_weights.get(name, 0.0) / active_sum for name in self.models}

    @classmethod
    def from_ensemble_config(cls, config: EnsembleConfig) -> "StrokePredictor":
        return cls(
            googlenet_checkpoint=config.googlenet_checkpoint,
            inception_v3_checkpoint=config.inception_v3_checkpoint,
            ensemble_weights=config.weights,
            device=config.device,
        )

    @torch.no_grad()
    def predict_probabilities(self, image_path: Path) -> tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        tensor = preprocess_ct_image(image_path, self.preprocess_config).unsqueeze(0).to(self.device)
        ensemble = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        per_model: Dict[str, Dict[str, float]] = {}

        for model_name, model in self.models.items():
            model.eval()
            logits = extract_logits(model(tensor))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            ensemble += self.weights[model_name] * probs
            per_model[model_name] = {class_name: float(probs[idx]) for idx, class_name in enumerate(CLASS_NAMES)}

        ensemble = ensemble / max(ensemble.sum(), 1e-12)
        ensemble = apply_decision_config(ensemble[None, :], self.decision_config)[0]
        return ensemble.astype(np.float32), per_model

    def predict_ct(
        self,
        image_path: Path,
        generate_explanations: bool = True,
        gradcam_class_index: Optional[int] = None,
    ) -> PredictionResult:
        image_path = Path(image_path)
        probs, per_model = self.predict_probabilities(image_path)
        predicted_index = int(np.argmax(probs))
        target_index = predicted_index if gradcam_class_index is None else gradcam_class_index
        gradcams: Dict[str, GradCAMResult] = {}
        if generate_explanations:
            for model_name, model in self.models.items():
                gradcams[model_name] = generate_gradcam(
                    model,
                    model_name,
                    image_path,
                    self.device,
                    class_index=target_index,
                    preprocess_config=self.preprocess_config,
                    target_resolution=self.gradcam_resolution,
                )

        return PredictionResult(
            predicted_class=CLASS_NAMES[predicted_index],
            predicted_index=predicted_index,
            confidence=float(probs[predicted_index]),
            probabilities={class_name: float(probs[idx]) for idx, class_name in enumerate(CLASS_NAMES)},
            model_probabilities=per_model,
            gradcams=gradcams,
        )


def predict_ct(
    image_path: str | Path,
    googlenet_checkpoint: str | Path = "D:/preprocessed/outputs/googlenet/best_model.pt",
    inception_v3_checkpoint: str | Path = "D:/preprocessed/outputs/inception_v3/best_model.pt",
    ensemble_weights: Sequence[float] = (0.4, 0.6),
    device: str = "auto",
    decision_config_path: str | Path | None = None,
    gradcam_resolution: str = "balanced",
) -> PredictionResult:
    decision_config = DecisionConfig.from_json(Path(decision_config_path)) if decision_config_path else None
    predictor = StrokePredictor(
        googlenet_checkpoint=Path(googlenet_checkpoint),
        inception_v3_checkpoint=Path(inception_v3_checkpoint),
        ensemble_weights=ensemble_weights,
        device=device,
        decision_config=decision_config,
        gradcam_resolution=gradcam_resolution,
    )
    return predictor.predict_ct(Path(image_path))
