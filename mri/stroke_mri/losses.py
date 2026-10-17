"""Loss functions specifically designed for imbalanced medical image classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedFocalLoss(nn.Module):
    """Class-weighted Focal Loss with optional label smoothing.

    Mitigates severe class imbalance by down-weighting well-classified easy examples
    (e.g., standard Normal scans) and focusing gradients on difficult stroke lesions.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
    ) -> None:
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 1 or logits.shape[-1] == 1:
            # Binary classification case
            targets = targets.float().view(-1, 1)
            probs = torch.sigmoid(logits.view(-1, 1))
            pt = probs * targets + (1.0 - probs) * (1.0 - targets)
            bce = F.binary_cross_entropy_with_logits(logits.view(-1, 1), targets, reduction="none")
            loss = ((1.0 - pt) ** self.gamma) * bce
            if self.alpha is not None:
                alpha_t = self.alpha[1] * targets + self.alpha[0] * (1.0 - targets)
                loss = alpha_t * loss
            return loss.mean()

        # Multiclass classification case
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def build_loss_function(
    loss_type: str = "focal",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.05,
) -> nn.Module:
    if loss_type.lower() == "focal":
        return ClassWeightedFocalLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            label_smoothing=label_smoothing,
        )
    if loss_type.lower() == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss type: {loss_type}. Choose 'focal' or 'cross_entropy'.")
