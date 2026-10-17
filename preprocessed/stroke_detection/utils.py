"""Utility helpers shared across the stroke CT project."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from .constants import CLASS_NAMES, CLASS_TO_IDX


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(payload: Dict[str, object], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def find_split_dir(data_root: Path, split: str) -> Path:
    """Find split folders in either clean or Google Drive extracted layouts."""
    direct = data_root / split
    if _looks_like_split(direct):
        return direct

    candidates = sorted(
        p for p in data_root.glob(f"{split}*") if p.is_dir()
    )
    for candidate in candidates:
        nested = candidate / split
        if _looks_like_split(nested):
            return nested
        if _looks_like_split(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find a valid '{split}' split under {data_root}. "
        f"Expected class folders: {CLASS_NAMES}."
    )


def _looks_like_split(path: Path) -> bool:
    return path.exists() and all((path / class_name).is_dir() for class_name in CLASS_NAMES)


def list_image_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def infer_patient_id(path: Path) -> str:
    """Best-effort patient id extraction for leakage checks.

    Medical datasets often encode slices as patient_slice.png, patient-001.png,
    or simply patient.png. This conservative heuristic catches common overlap.
    """
    stem = path.stem.lower()
    for token in ["_slice", "-slice", "_scan", "-scan", "_img", "-img"]:
        if token in stem:
            return stem.split(token)[0]
    parts = stem.replace("-", "_").split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def check_patient_leakage(split_to_paths: Dict[str, Sequence[Path]]) -> Dict[str, List[str]]:
    split_to_patients = {
        split: {infer_patient_id(path) for path in paths}
        for split, paths in split_to_paths.items()
    }
    overlaps: Dict[str, List[str]] = {}
    splits = list(split_to_patients)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(split_to_patients[left] & split_to_patients[right])
            if overlap:
                overlaps[f"{left}_vs_{right}"] = overlap
    return overlaps


def class_counts(samples: Iterable[tuple[Path, int]]) -> Dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for _, label in samples:
        counts[CLASS_NAMES[label]] += 1
    return counts


def validate_class_mapping(split_dir: Path) -> None:
    actual = {p.name for p in split_dir.iterdir() if p.is_dir()}
    expected = set(CLASS_TO_IDX)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"Missing class folders in {split_dir}: {missing}")
