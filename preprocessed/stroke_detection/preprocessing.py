"""CT image loading and preprocessing used identically for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = IMAGE_SIZE
    clahe_clip_limit: float = 1.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    enable_very_light_denoising: bool = False
    denoise_h: int = 3


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable or corrupted image: {path}")
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def mild_clahe(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid_size,
    )
    return clahe.apply(gray)


def very_light_denoise(gray: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    """Optional conservative denoising.

    This is intentionally disabled by default because subtle ischemic findings can
    present as faint low-contrast texture changes.
    """
    return cv2.fastNlMeansDenoising(gray, None, h=config.denoise_h, templateWindowSize=3, searchWindowSize=7)


def preprocess_ct_image(
    image_path: Path,
    config: Optional[PreprocessConfig] = None,
    return_original_rgb: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, np.ndarray]:
    config = config or PreprocessConfig()
    gray = load_grayscale_image(image_path)
    gray = cv2.resize(gray, (config.image_size, config.image_size), interpolation=cv2.INTER_AREA)
    gray = mild_clahe(gray, config)
    if config.enable_very_light_denoising:
        gray = very_light_denoise(gray, config)

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = (tensor - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(IMAGENET_STD, dtype=np.float32)
    tensor = torch.from_numpy(tensor).permute(2, 0, 1).contiguous()

    if return_original_rgb:
        return tensor, rgb
    return tensor


def tensor_to_display_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Unnormalize an ImageNet-normalized tensor for visualization."""
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = image * np.asarray(IMAGENET_STD) + np.asarray(IMAGENET_MEAN)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255).astype(np.uint8)
