"""Model construction, checkpointing, and fine-tuning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision import models

from .constants import CLASS_TO_IDX


import sys

# Ensure stroke_mri is discoverable for hybrid model sharing
MRI_DIR = Path(__file__).resolve().parents[2] / "mri"
if str(MRI_DIR) not in sys.path and MRI_DIR.exists():
    sys.path.insert(0, str(MRI_DIR))

SUPPORTED_MODELS = ("googlenet", "inception_v3", "hybrid_cbam_swin")


def build_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = True,
) -> nn.Module:
    model_name = normalize_model_name(model_name)
    if model_name == "googlenet":
        model = _build_googlenet(num_classes, pretrained, dropout)
    elif model_name == "inception_v3":
        model = _build_inception_v3(num_classes, pretrained, dropout)
    elif model_name == "hybrid_cbam_swin":
        from stroke_mri.hybrid_model import build_hybrid_model, freeze_hybrid_backbones
        model = build_hybrid_model(num_classes, pretrained=pretrained, dropout=dropout)
        if freeze_backbone:
            freeze_hybrid_backbones(model)
        return model
    else:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {SUPPORTED_MODELS}.")

    if freeze_backbone:
        freeze_all_except_classifier(model, model_name)
    return model


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "_")
    if normalized in {"inception", "inceptionv3"}:
        return "inception_v3"
    if normalized in {"google_net", "googlenet"}:
        return "googlenet"
    if normalized in {"hybrid", "cbam_swin", "hybrid_cbam_swin", "hybrid_swin"}:
        return "hybrid_cbam_swin"
    return normalized



def _build_googlenet(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.googlenet(weights=weights, aux_logits=True, transform_input=False, init_weights=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    if model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    if model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
    return model


def _build_inception_v3(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.inception_v3(weights=weights, aux_logits=True, transform_input=False, init_weights=False)
    if model.AuxLogits is not None:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model


def freeze_all_except_classifier(model: nn.Module, model_name: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for module_name in classifier_module_names(model_name):
        module = getattr(model, module_name, None)
        if module is not None:
            for param in module.parameters():
                param.requires_grad = True


def unfreeze_deep_layers(model: nn.Module, model_name: str) -> None:
    """Fine-tune later high-level feature blocks while keeping early edges stable."""
    model_name = normalize_model_name(model_name)
    if model_name == "hybrid_cbam_swin":
        from stroke_mri.hybrid_model import unfreeze_hybrid_backbones
        unfreeze_hybrid_backbones(model)
        return
    trainable_prefixes = {
        "googlenet": ("inception4e", "inception5a", "inception5b", "aux1", "aux2", "fc"),
        "inception_v3": ("Mixed_7a", "Mixed_7b", "Mixed_7c", "AuxLogits", "fc"),
    }[model_name]
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(trainable_prefixes)


def unfreeze_all_layers(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def classifier_module_names(model_name: str) -> tuple[str, ...]:
    model_name = normalize_model_name(model_name)
    if model_name == "googlenet":
        return ("aux1", "aux2", "fc")
    if model_name == "inception_v3":
        return ("AuxLogits", "fc")
    if model_name == "hybrid_cbam_swin":
        return ("cbam3", "cbam4", "fusion", "classifier")
    raise ValueError(f"Unsupported model '{model_name}'.")


def get_gradcam_target_layer(model: nn.Module, model_name: str, resolution: str = "balanced") -> nn.Module:
    """Return a convolutional block for Grad-CAM."""
    model_name = normalize_model_name(model_name)
    if model_name == "hybrid_cbam_swin":
        return model.gradcam_target_layer

    resolution = resolution.lower()
    if model_name == "googlenet":
        if resolution == "final":
            return model.inception5b
        if resolution == "high":
            return model.inception4d
        return model.inception4e
    if model_name == "inception_v3":
        if resolution == "final":
            return model.Mixed_7c
        if resolution == "high":
            return model.Mixed_6d
        return model.Mixed_6e
    raise ValueError(f"Unsupported model '{model_name}'.")


def extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    raise TypeError(f"Cannot extract logits from output type {type(outputs)}")


def compute_auxiliary_loss(outputs: Any, labels: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    logits = extract_logits(outputs)
    loss = criterion(logits, labels)
    aux_terms = []
    if hasattr(outputs, "aux_logits") and outputs.aux_logits is not None:
        aux_terms.append(outputs.aux_logits)
    if hasattr(outputs, "aux_logits1") and outputs.aux_logits1 is not None:
        aux_terms.append(outputs.aux_logits1)
    if hasattr(outputs, "aux_logits2") and outputs.aux_logits2 is not None:
        aux_terms.append(outputs.aux_logits2)
    for aux_logits in aux_terms:
        loss = loss + 0.3 * criterion(aux_logits, labels)
    return loss


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    model_name: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": normalize_model_name(model_name),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "metrics": metrics,
            "config": config,
            "class_to_idx": CLASS_TO_IDX,
        },
        path,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    dropout: float = 0.5,
) -> tuple[nn.Module, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_to_idx = checkpoint.get("class_to_idx")
    if class_to_idx != CLASS_TO_IDX:
        raise ValueError(
            f"Checkpoint class mapping {class_to_idx} does not match required mapping {CLASS_TO_IDX}."
        )
    model_name = normalize_model_name(checkpoint["model_name"])
    model = build_model(model_name, pretrained=False, dropout=dropout, freeze_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
