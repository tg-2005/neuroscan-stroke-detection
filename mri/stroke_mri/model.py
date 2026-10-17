from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNet50StrokeClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.5, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)
        in_features = self.backbone.fc.in_features
        output_dim = 1 if num_classes == 2 else num_classes
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, output_dim),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.backbone.layer4[-1]


def build_model(num_classes: int, dropout: float = 0.5, pretrained: bool = True) -> ResNet50StrokeClassifier:
    return ResNet50StrokeClassifier(num_classes=num_classes, dropout=dropout, pretrained=pretrained)


def freeze_backbone(model: ResNet50StrokeClassifier) -> None:
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = name.startswith("fc")


def unfreeze_deep_layers(model: ResNet50StrokeClassifier) -> None:
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc")
