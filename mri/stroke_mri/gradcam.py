from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from .preprocessing import PreprocessConfig, preprocess_for_inference


class GradCAM:
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: torch.nn.Module,
        class_names: list[str] | None = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.class_names = class_names
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, _module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _backward_hook(
        self,
        _module: torch.nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __call__(self, image_tensor: torch.Tensor, target_index: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if logits.shape[1] == 1:
            normal_index = self.class_names.index("Normal") if self.class_names and "Normal" in self.class_names else 0
            if target_index == normal_index:
                score = -logits[:, 0]
            else:
                score = logits[:, 0]
        else:
            if target_index is None:
                target_index = int(torch.argmax(logits, dim=1).item())
            score = logits[:, target_index]
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        activations = self.activations[0]
        gradients = self.gradients[0]
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0))
        cam_np = cam.detach().cpu().numpy()
        cam_np -= cam_np.min()
        cam_np /= cam_np.max() + 1e-8
        return cam_np


def make_heatmap_overlay(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.38,
    threshold_background: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    cam_resized = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    if threshold_background:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        brain_mask = gray > max(5, np.percentile(gray, 10))
        cam_resized = cam_resized * brain_mask.astype(np.float32)
    heatmap = (cam_resized * 255).clip(0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_color, alpha, 0)
    return heatmap_color, overlay


def save_gradcam_panel(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    out_path: str | Path,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, image, label in zip(axes, [original_rgb, heatmap_rgb, overlay_rgb], ["Original", "Grad-CAM", "Overlay"]):
        ax.imshow(image)
        ax.set_title(label)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def gradcam_for_path(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    image_path: str | Path,
    preprocess_config: PreprocessConfig,
    device: torch.device,
    target_index: int | None = None,
    class_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor, processed = preprocess_for_inference(image_path, preprocess_config)
    tensor = tensor.unsqueeze(0).to(device)
    cam_engine = GradCAM(model, target_layer, class_names=class_names)
    try:
        cam = cam_engine(tensor, target_index=target_index)
    finally:
        cam_engine.remove()
    heatmap, overlay = make_heatmap_overlay(processed, cam)
    return processed, heatmap, overlay
