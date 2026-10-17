"""Loss functions for imbalanced medical image classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Class-weighted focal loss.

    Useful when the model learns easy Normal examples faster than subtle
    ischemia. Gamma focuses learning on hard or misclassified samples.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = (1.0 - pt).pow(self.gamma) * ce
        return focal.mean()


def build_loss(
    loss_name: str,
    class_weights: Optional[torch.Tensor],
    focal_gamma: float,
    label_smoothing: float,
) -> nn.Module:
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss '{loss_name}'. Use 'cross_entropy' or 'focal'.")
