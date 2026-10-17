"""Cross-Attention Feature Fusion Module (CAFM) for Hybrid CNN-Transformer networks.

Allows global ViT / Swin queries to attend to high-resolution local CNN+CBAM spatial tokens,
providing an attention-weighted fusion that resists overfitting on small cohorts.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusionModule(nn.Module):
    """Multi-Head Cross-Attention bridging Swin-Transformer (queries) and ResNet-CBAM (keys/values)."""

    def __init__(
        self,
        cnn_dim: int = 2048,
        transformer_dim: int = 768,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Projections to shared embedding space
        self.q_proj = nn.Linear(transformer_dim, embed_dim)
        self.k_proj = nn.Linear(cnn_dim, embed_dim)
        self.v_proj = nn.Linear(cnn_dim, embed_dim)

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Bottleneck Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cnn_features: torch.Tensor,
        transformer_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Args:

            cnn_features: [B, C_cnn, H, W] local spatial feature map from ResNet+CBAM.
            transformer_features: [B, C_trans] or [B, N_tokens, C_trans] global representation.

        Returns:
            fused_features: [B, embed_dim] bottleneck feature vector.
            attn_weights: [B, N_q, N_k] cross-attention map for explainability.
        """
        b, c, h, w = cnn_features.shape

        # Flatten spatial dimensions into a sequence of spatial tokens: [B, H*W, C_cnn]
        cnn_seq = cnn_features.view(b, c, h * w).permute(0, 2, 1)

        # Ensure transformer features have a token dimension: [B, 1, C_trans] if pooled
        if transformer_features.ndim == 2:
            trans_seq = transformer_features.unsqueeze(1)
        else:
            trans_seq = transformer_features

        # Project to common embedding dimension
        q = self.q_proj(trans_seq)  # [B, 1, embed_dim]
        k = self.k_proj(cnn_seq)    # [B, H*W, embed_dim]
        v = self.v_proj(cnn_seq)    # [B, H*W, embed_dim]

        # Multi-head cross attention: Global queries attend to local key-value tokens
        attn_out, attn_weights = self.multihead_attn(q, k, v)
        x = self.norm1(q + attn_out)

        # Feed-forward block
        x = self.norm2(x + self.ffn(x))

        # Mean-pool over sequence dimension to obtain a fixed-size vector: [B, embed_dim]
        fused = x.mean(dim=1)
        return fused, attn_weights


class BottleneckClassifierHead(nn.Module):
    """Regularized classification head on top of the cross-attention bottleneck."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.4) -> None:
        super().__init__()
        output_dim = 1 if num_classes == 2 else num_classes
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
