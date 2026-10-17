"""Dataset and DataLoader utilities for preprocessed CT scans."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .constants import CLASS_NAMES, CLASS_TO_IDX
from .preprocessing import PreprocessConfig, preprocess_ct_image
from .utils import (
    check_patient_leakage,
    class_counts,
    find_split_dir,
    list_image_files,
    validate_class_mapping,
)


class MedicalAugmenter:
    """Conservative CT-safe augmentation for training only."""

    def __init__(
        self,
        rotation_degrees: float = 10.0,
        horizontal_flip_p: float = 0.5,
        brightness_delta: float = 0.08,
        zoom_range: float = 0.10,
    ) -> None:
        self.rotation_degrees = rotation_degrees
        self.horizontal_flip_p = horizontal_flip_p
        self.brightness_delta = brightness_delta
        self.zoom_range = zoom_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        image = tensor.permute(1, 2, 0).numpy().copy()
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        image = np.clip(image * std + mean, 0.0, 1.0)

        h, w = image.shape[:2]
        if random.random() < self.horizontal_flip_p:
            image = cv2.flip(image, 1)

        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        zoom = random.uniform(1.0 - self.zoom_range, 1.0 + self.zoom_range)
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, zoom)
        image = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        brightness = random.uniform(-self.brightness_delta, self.brightness_delta)
        image = np.clip(image + brightness, 0.0, 1.0)

        image = (image - mean) / std
        return torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).contiguous()


class StrokeCTDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        preprocess_config: Optional[PreprocessConfig] = None,
        augment: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        validate_images: bool = True,
    ) -> None:
        self.split_dir = Path(split_dir)
        validate_class_mapping(self.split_dir)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.augment = augment
        self.samples = self._collect_samples(validate_images=validate_images)
        if not self.samples:
            raise ValueError(f"No valid images found in {self.split_dir}")

    def _collect_samples(self, validate_images: bool) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []
        for class_name in CLASS_NAMES:
            class_dir = self.split_dir / class_name
            for image_path in list_image_files(class_dir):
                if validate_images and cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE) is None:
                    print(f"[WARN] Skipping unreadable image: {image_path}")
                    continue
                samples.append((image_path, CLASS_TO_IDX[class_name]))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        image_path, label = self.samples[index]
        image = preprocess_ct_image(image_path, self.preprocess_config)
        if self.augment is not None:
            image = self.augment(image)
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(image_path),
        }


def build_datasets(
    data_root: Path,
    preprocess_config: Optional[PreprocessConfig] = None,
    validate_images: bool = True,
) -> Dict[str, StrokeCTDataset]:
    split_dirs = {split: find_split_dir(Path(data_root), split) for split in ["train", "val", "test"]}
    datasets = {
        "train": StrokeCTDataset(
            split_dirs["train"],
            preprocess_config=preprocess_config,
            augment=MedicalAugmenter(),
            validate_images=validate_images,
        ),
        "val": StrokeCTDataset(split_dirs["val"], preprocess_config=preprocess_config, validate_images=validate_images),
        "test": StrokeCTDataset(split_dirs["test"], preprocess_config=preprocess_config, validate_images=validate_images),
    }

    leakage = check_patient_leakage({split: [p for p, _ in ds.samples] for split, ds in datasets.items()})
    if leakage:
        preview = {key: values[:10] for key, values in leakage.items()}
        raise RuntimeError(
            "Potential patient/image leakage detected between splits. "
            f"Examples: {preview}. Fix the split before training."
        )
    return datasets


def compute_class_weights(
    samples: Sequence[Tuple[Path, int]],
    ischemia_boost: float = 1.25,
) -> torch.Tensor:
    labels = [label for _, label in samples]
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    weights[CLASS_TO_IDX["Ischemia"]] *= ischemia_boost
    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(samples: Sequence[Tuple[Path, int]], class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = [float(class_weights[label]) for _, label in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def build_dataloaders(
    datasets: Dict[str, StrokeCTDataset],
    batch_size: int,
    num_workers: int,
    use_weighted_sampler: bool,
    ischemia_loss_boost: float,
) -> tuple[Dict[str, DataLoader], torch.Tensor, Dict[str, Dict[str, int]]]:
    class_weight_tensor = compute_class_weights(datasets["train"].samples, ischemia_boost=ischemia_loss_boost)
    sampler = build_sampler(datasets["train"].samples, class_weight_tensor) if use_weighted_sampler else None

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        ),
    }
    counts = {split: class_counts(ds.samples) for split, ds in datasets.items()}
    return loaders, class_weight_tensor, counts
