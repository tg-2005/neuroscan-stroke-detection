"""Grad-CAM explainability for GoogleNet and InceptionV3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .constants import CLASS_NAMES
from .models import extract_logits, get_gradcam_target_layer
from .preprocessing import PreprocessConfig, preprocess_ct_image


@dataclass
class GradCAMResult:
    heatmap: np.ndarray
    heatmap_rgb: np.ndarray
    overlay: np.ndarray
    original_rgb: np.ndarray
    class_index: int
    target_resolution: str


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module: nn.Module, _input: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, input_tensor: torch.Tensor, class_index: Optional[int] = None) -> GradCAMResult:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(input_tensor)
        logits = extract_logits(outputs)
        if class_index is None:
            class_index = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_index].sum()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        gradients = self.gradients[0]
        activations = self.activations[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0))
        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[-1], input_tensor.shape[-2]), interpolation=cv2.INTER_CUBIC)
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 1e-8:
            cam = cam / denom
        cam = enhance_heatmap_contrast(cam)
        return GradCAMResult(
            heatmap=cam.astype(np.float32),
            heatmap_rgb=np.empty((0, 0, 3), dtype=np.uint8),
            overlay=np.empty((0, 0, 3), dtype=np.uint8),
            original_rgb=np.empty((0, 0, 3), dtype=np.uint8),
            class_index=class_index,
            target_resolution="",
        )


def enhance_heatmap_contrast(heatmap: np.ndarray) -> np.ndarray:
    """Normalize by robust percentiles so focal CT evidence is not washed out."""
    heatmap = np.clip(heatmap, 0.0, 1.0)
    high = float(np.percentile(heatmap, 99.0))
    low = float(np.percentile(heatmap, 5.0))
    if high > low:
        heatmap = np.clip((heatmap - low) / (high - low), 0.0, 1.0)
    return heatmap


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    color_map = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)


def overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    color_map = colorize_heatmap(heatmap)
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, color_map, alpha, 0)
    return overlay


def generate_gradcam(
    model: nn.Module,
    model_name: str,
    image_path: Path,
    device: torch.device,
    class_index: Optional[int] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    target_resolution: str = "balanced",
) -> GradCAMResult:
    tensor, original_rgb = preprocess_ct_image(
        image_path,
        preprocess_config or PreprocessConfig(),
        return_original_rgb=True,
    )
    input_tensor = tensor.unsqueeze(0).to(device)
    was_training = model.training
    model.eval()
    target_layer = get_gradcam_target_layer(model, model_name, resolution=target_resolution)
    gradcam = GradCAM(model, target_layer)
    try:
        result = gradcam(input_tensor, class_index=class_index)
    finally:
        gradcam.remove_hooks()
        model.train(was_training)

    result.original_rgb = original_rgb
    result.heatmap_rgb = colorize_heatmap(result.heatmap)
    result.overlay = overlay_heatmap(original_rgb, result.heatmap)
    result.target_resolution = target_resolution
    return result


def save_gradcam_figure(
    result: GradCAMResult,
    output_path: Path,
    title: str = "",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(result.original_rgb, cmap="gray")
    axes[0].set_title("Original CT")
    axes[1].imshow(result.heatmap_rgb)
    axes[1].set_title(f"Grad-CAM: {CLASS_NAMES[result.class_index]}")
    axes[2].imshow(result.overlay)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
