"""Hybrid CNN-Transformer architecture combining ResNet50+CBAM and Swin-Transformer.

Fuses fine-grained local lesion textures with brain-wide cross-hemispheric contextual attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, Swin_T_Weights, resnet50, swin_t

from .cbam import CBAM
from .fusion import BottleneckClassifierHead, CrossAttentionFusionModule


class HybridStrokeClassifier(nn.Module):
    """Hybrid Network: ResNet50 with CBAM attention + Swin-Transformer with Cross-Attention Fusion."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.4,
        embed_dim: int = 256,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # 1. Local CNN Branch (ResNet50 + CBAM)
        res_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        base_res = resnet50(weights=res_weights)

        self.conv1 = base_res.conv1
        self.bn1 = base_res.bn1
        self.relu = base_res.relu
        self.maxpool = base_res.maxpool

        self.layer1 = base_res.layer1
        self.layer2 = base_res.layer2
        self.layer3 = base_res.layer3
        self.cbam3 = CBAM(in_planes=1024, ratio=16)

        self.layer4 = base_res.layer4
        self.cbam4 = CBAM(in_planes=2048, ratio=16)

        # 2. Global Transformer Branch (Swin-T)
        swin_weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        self.swin = swin_t(weights=swin_weights)

        # 3. Cross-Attention Feature Fusion Module (CAFM)
        self.fusion = CrossAttentionFusionModule(
            cnn_dim=2048,
            transformer_dim=768,
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 4. Bottleneck Classifier
        self.classifier = BottleneckClassifierHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        self._last_attn_weights: torch.Tensor | None = None

    def extract_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.cbam3(out)

        out = self.layer4(out)
        out = self.cbam4(out)
        return out

    def extract_swin_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.swin.features(x)
        features = self.swin.norm(features)
        features = self.swin.permute(features)
        features = self.swin.avgpool(features)
        return torch.flatten(features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract dual streams
        cnn_feat = self.extract_cnn_features(x)       # [B, 2048, 7, 7]
        swin_feat = self.extract_swin_features(x)     # [B, 768]

        # Cross-attention fusion
        fused, attn_weights = self.fusion(cnn_feat, swin_feat)
        self._last_attn_weights = attn_weights

        # Classification logits
        logits = self.classifier(fused)
        return logits.squeeze(-1) if self.num_classes == 2 else logits

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Target layer for Grad-CAM explainability (CBAM on layer4)."""
        return self.cbam4

    @property
    def last_cross_attention_weights(self) -> torch.Tensor | None:
        """Returns the last computed cross-attention matrix for visual rollout."""
        return self._last_attn_weights


def build_hybrid_model(
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.4,
    embed_dim: int = 256,
) -> HybridStrokeClassifier:
    return HybridStrokeClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        embed_dim=embed_dim,
    )


def freeze_hybrid_backbones(model: HybridStrokeClassifier) -> None:
    """Freezes both ResNet and Swin backbones, training only CBAM, CAFM, and classifier."""
    for param in model.conv1.parameters():
        param.requires_grad = False
    for param in model.bn1.parameters():
        param.requires_grad = False
    for layer in (model.layer1, model.layer2, model.layer3, model.layer4):
        for param in layer.parameters():
            param.requires_grad = False
    for param in model.swin.parameters():
        param.requires_grad = False

    # Keep CBAM, CAFM, and Classifier trainable
    for param in model.cbam3.parameters():
        param.requires_grad = True
    for param in model.cbam4.parameters():
        param.requires_grad = True
    for param in model.fusion.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_hybrid_backbones(model: HybridStrokeClassifier) -> None:
    """Unfreezes deep layers of both backbones for fine-tuning."""
    for layer in (model.layer3, model.layer4):
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.swin.features[-2:].parameters():
        param.requires_grad = True
