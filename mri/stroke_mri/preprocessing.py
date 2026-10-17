from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    apply_clahe: bool = True
    skull_strip: bool = True
    denoise: bool = False
    preserve_aspect_ratio: bool = True
    enhance_bilateral_asymmetry: bool = False


def _read_dicom(path: Path) -> np.ndarray:
    import pydicom

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr
    low, high = np.percentile(arr, [0.5, 99.5])
    if high > low:
        arr = np.clip(arr, low, high)
    arr -= np.min(arr)
    denom = np.max(arr) + 1e-8
    arr = (arr / denom * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def load_mri_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix in {".dcm", ".dicom"}:
            image = _read_dicom(path)
        else:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                image = np.array(img)
    except Exception as exc:
        raise ValueError(f"Unreadable MRI image: {path}") from exc

    if image is None or image.size == 0:
        raise ValueError(f"Empty MRI image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif image.shape[2] != 3:
            raise ValueError(f"Unsupported channel count {image.shape[2]} for {path}")
    else:
        raise ValueError(f"Unsupported image dimensions {image.shape} for {path}")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image


def light_skull_strip(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels <= 1:
        return image_rgb
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    brain_mask = (labels == largest).astype(np.uint8) * 255
    area_ratio = float(np.count_nonzero(brain_mask)) / float(brain_mask.size)
    if area_ratio < 0.10 or area_ratio > 0.95:
        return image_rgb
    brain_mask = cv2.dilate(brain_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.bitwise_and(image_rgb, image_rgb, mask=brain_mask)


def apply_clahe_rgb(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def resize_mri(image_rgb: np.ndarray, image_size: int, preserve_aspect_ratio: bool = True) -> np.ndarray:
    if not preserve_aspect_ratio:
        return cv2.resize(image_rgb, (image_size, image_size), interpolation=cv2.INTER_AREA)
    h, w = image_rgb.shape[:2]
    scale = min(image_size / h, image_size / w)
    new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    top = (image_size - new_h) // 2
    left = (image_size - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def compute_bilateral_asymmetry_map(image_rgb: np.ndarray) -> np.ndarray:
    """Computes left-right hemispheric absolute difference to highlight stroke lesions.

    Normal brain parenchyma is largely symmetrical across the sagittal midline.
    Acute ischemic infarcts (edema) and intracranial hemorrhages create prominent
    cross-hemispheric intensity disparities.
    """
    flipped = cv2.flip(image_rgb, 1)
    diff = cv2.absdiff(image_rgb, flipped)
    return diff


def preprocess_numpy(image_rgb: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    image = image_rgb.copy()
    if config.skull_strip:
        image = light_skull_strip(image)
    if config.apply_clahe:
        image = apply_clahe_rgb(image)
    if config.denoise:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    resized = resize_mri(image, config.image_size, config.preserve_aspect_ratio)
    if config.enhance_bilateral_asymmetry:
        diff = compute_bilateral_asymmetry_map(resized)
        # Subtle alpha blend (15% asymmetry boost) to highlight lesion disparity
        resized = cv2.addWeighted(resized, 0.85, diff, 0.15, 0)
    return resized


def to_tensor_normalized(image_rgb: np.ndarray) -> torch.Tensor:
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float()


def preprocess_for_inference(path: str | Path, config: PreprocessConfig) -> tuple[torch.Tensor, np.ndarray]:
    original = load_mri_image(path)
    processed = preprocess_numpy(original, config)
    return to_tensor_normalized(processed), processed
