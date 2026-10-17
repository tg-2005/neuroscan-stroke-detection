"""Convolutional Block Attention Module (CBAM) for medical image feature refinement.

Paper: "CBAM: Convolutional Block Attention Module" (Woo et al., ECCV 2018)
Adapted for brain lesion focus (ischemia / hemorrhage contrast vs. normal background).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention mechanism to capture 'what' features are meaningful."""

    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        hidden_planes = max(1, in_planes // ratio)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, hidden_planes, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_planes, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to capture 'where' lesions/infarcts are located."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """Composite Channel and Spatial Attention block with residual connection."""

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out + x
