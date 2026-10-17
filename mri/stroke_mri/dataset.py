from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from .preprocessing import PreprocessConfig, load_mri_image, preprocess_numpy, to_tensor_normalized

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm", ".dicom"}
NORMAL_NAMES = {"normal", "healthy", "no_stroke", "nostroke", "control"}
ISCHEMIC_NAMES = {"ischemic", "ischaemic", "ischemic_stroke", "ischaemic_stroke", "infarct"}
HEMORRHAGIC_NAMES = {"hemorrhagic", "haemorrhagic", "hemorrhage", "haemorrhage", "bleed", "ich"}
GENERATED_AUGMENT_PREFIXES = ("aug_", "std_aug_", "train_aug_", "flip_", "rot_", "copy_")
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "site-packages",
    "manifests",
    "manifests_binary",
    "manifests_smoke",
    "outputs",
}
BINARY_CLASS_ORDER = ["Normal", "Stroke"]
MULTICLASS_CLASS_ORDER = ["Ischemic", "Hemorrhagic", "Normal"]


@dataclass(frozen=True)
class ClassSpec:
    class_names: list[str]
    class_to_idx: dict[str, int]
    mode: str


def infer_patient_id(path: str | Path) -> str:
    stem = Path(path).stem.lower()
    stem = re.sub(r"^(aug|std_aug|train_aug|flip|rot|copy)[_-]*", "", stem)
    stem = re.sub(r"(_aug|_copy|_flip|_rot)\d*$", "", stem)
    match = re.search(r"([a-z]+[_-]?\d+)", stem)
    return match.group(1) if match else stem


def _is_excluded_path(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        relative_parts = path.parts
    return any(part in EXCLUDED_DIR_NAMES for part in relative_parts)


def _canonical_label(label: str, class_mode: str) -> str:
    normalized = label.strip().lower().replace(" ", "_")
    if class_mode == "binary":
        return "Normal" if normalized in NORMAL_NAMES else "Stroke"
    if normalized in ISCHEMIC_NAMES:
        return "Ischemic"
    if normalized in HEMORRHAGIC_NAMES:
        return "Hemorrhagic"
    if normalized in NORMAL_NAMES:
        return "Normal"
    return label.strip()


def is_generated_augmentation(path: str | Path) -> bool:
    return Path(path).stem.lower().startswith(GENERATED_AUGMENT_PREFIXES)


def resolve_class_spec(labels: Iterable[str], class_mode: str = "auto") -> ClassSpec:
    raw = sorted({str(label) for label in labels})
    if class_mode == "auto":
        non_normal = [x for x in raw if x.lower().replace(" ", "_") not in NORMAL_NAMES]
        class_mode = "binary" if len(non_normal) == 1 else "multiclass"
    mapped_set = {_canonical_label(label, class_mode) for label in raw}
    if class_mode == "binary":
        mapped = [name for name in BINARY_CLASS_ORDER if name in mapped_set]
    else:
        ordered = [name for name in MULTICLASS_CLASS_ORDER if name in mapped_set]
        extras = sorted(mapped_set - set(MULTICLASS_CLASS_ORDER))
        mapped = ordered + extras
    class_to_idx = {name: idx for idx, name in enumerate(mapped)}
    return ClassSpec(mapped, class_to_idx, class_mode)


def discover_images(
    data_root: str | Path,
    class_mode: str = "auto",
    include_generated_augmented: bool = False,
) -> tuple[pd.DataFrame, ClassSpec]:
    root = Path(data_root)
    records: list[dict[str, str]] = []
    split_dirs = [root / split for split in ("train", "val", "test")]
    has_splits = all(path.exists() for path in split_dirs)
    search_roots = split_dirs if has_splits else [root]

    for search_root in search_roots:
        split = search_root.name if has_splits else ""
        for class_dir in sorted([p for p in search_root.iterdir() if p.is_dir()]):
            if _is_excluded_path(class_dir, root):
                continue
            for image_path in class_dir.rglob("*"):
                if _is_excluded_path(image_path, root):
                    continue
                if image_path.suffix.lower() in IMAGE_EXTENSIONS and image_path.is_file():
                    if is_generated_augmentation(image_path) and not include_generated_augmented:
                        continue
                    records.append(
                        {
                            "path": str(image_path.resolve()),
                            "source_label": class_dir.name,
                            "split": split,
                            "patient_id": infer_patient_id(image_path),
                        }
                    )
    if not records:
        raise ValueError(f"No MRI images found under {root}")

    df = pd.DataFrame(records)
    spec = resolve_class_spec(df["source_label"], class_mode)
    df["label"] = df["source_label"].map(lambda x: _canonical_label(x, spec.mode))
    df["target"] = df["label"].map(spec.class_to_idx)
    return df, spec


def validate_readable_images(df: pd.DataFrame) -> pd.DataFrame:
    valid_rows = []
    for row in df.itertuples(index=False):
        try:
            _ = load_mri_image(row.path)
            valid_rows.append(row._asdict())
        except Exception as exc:
            print(f"[WARN] Skipping unreadable image: {row.path} ({exc})")
    if not valid_rows:
        raise ValueError("All images were unreadable.")
    return pd.DataFrame(valid_rows)


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leakage_report(df: pd.DataFrame) -> dict[str, object]:
    report: dict[str, object] = {"patient_overlap": {}, "duplicate_hash_overlap": []}
    if "split" not in df.columns or df["split"].eq("").all():
        return report
    for a in sorted(df["split"].unique()):
        for b in sorted(df["split"].unique()):
            if a >= b:
                continue
            pa = set(df.loc[df["split"] == a, "patient_id"])
            pb = set(df.loc[df["split"] == b, "patient_id"])
            overlap = sorted(pa & pb)
            if overlap:
                report["patient_overlap"][f"{a}-{b}"] = overlap[:50]

    hashed = df.copy()
    hashed["sha256"] = hashed["path"].map(file_sha256)
    for digest, group in hashed.groupby("sha256"):
        splits = sorted(set(group["split"]))
        if len(splits) > 1:
            report["duplicate_hash_overlap"].append(
                {"sha256": digest, "splits": splits, "paths": group["path"].head(10).tolist()}
            )
    return report


def attach_duplicate_group_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hashes = []
    for path in df["path"]:
        try:
            hashes.append(file_sha256(path))
        except OSError:
            hashes.append(f"unreadable::{path}")
    df["sha256"] = hashes

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in df.itertuples(index=False):
        patient_node = f"patient::{row.patient_id}"
        hash_node = f"hash::{row.sha256}"
        union(patient_node, hash_node)

    df["group_id"] = [
        find(f"patient::{patient_id}") for patient_id in df["patient_id"].astype(str)
    ]
    return df


def create_grouped_manifests(
    data_root: str | Path,
    out_dir: str | Path,
    class_mode: str = "auto",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    validate_images: bool = True,
    include_generated_augmented: bool = False,
) -> tuple[dict[str, Path], ClassSpec]:
    df, spec = discover_images(data_root, class_mode, include_generated_augmented=include_generated_augmented)
    if df["split"].ne("").any():
        raise ValueError("Input already appears to contain train/val/test folders; manifests are not needed.")
    if validate_images:
        df = validate_readable_images(df)
    df = attach_duplicate_group_ids(df)

    groups = df["group_id"].to_numpy()
    train_idx, temp_idx = next(GroupShuffleSplit(n_splits=1, test_size=val_size + test_size, random_state=seed).split(df, df["target"], groups))
    temp = df.iloc[temp_idx].reset_index(drop=True)
    relative_test = test_size / (val_size + test_size)
    val_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=seed + 1).split(
            temp, temp["target"], temp["group_id"]
        )
    )
    splits = {
        "train": df.iloc[train_idx].copy(),
        "val": temp.iloc[val_idx].copy(),
        "test": temp.iloc[test_idx].copy(),
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, split_df in splits.items():
        split_df["split"] = split
        path = out_dir / f"{split}.csv"
        split_df.to_csv(path, index=False)
        paths[split] = path

    all_splits = pd.concat(splits.values(), ignore_index=True)
    report = leakage_report(all_splits)
    if report["patient_overlap"] or report["duplicate_hash_overlap"]:
        raise RuntimeError(f"Data leakage detected after splitting: {report}")
    return paths, spec


class MRIDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        preprocess_config: PreprocessConfig,
        augment: bool = False,
    ) -> None:
        self.df = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.reset_index(drop=True)
        self.config = preprocess_config
        self.augment = augment
        self.augmenter = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomRotation(degrees=10, fill=0),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.10, contrast=0.10),
                transforms.RandomAffine(degrees=0, scale=(0.95, 1.05), fill=0),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, object] | None:
        row = self.df.iloc[index]
        try:
            image = load_mri_image(row["path"])
            image = preprocess_numpy(image, self.config)
            if self.augment:
                image = np.array(self.augmenter(image))
            tensor = to_tensor_normalized(image)
            return {
                "image": tensor,
                "target": torch.tensor(int(row["target"]), dtype=torch.long),
                "path": row["path"],
                "label": row["label"],
            }
        except Exception as exc:
            print(f"[WARN] Skipping unreadable image at runtime: {row['path']} ({exc})")
            return None


def make_weighted_sampler(df: pd.DataFrame, power: float = 0.75) -> WeightedRandomSampler:
    counts = df["target"].value_counts().to_dict()
    # Smoothed inverse class frequency: balances batches while preventing single-sample overfitting
    weights = df["target"].map(lambda t: 1.0 / (counts[int(t)] ** power)).to_numpy(dtype=np.float64)
    return WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)


def collate_skip_bad(batch: list[dict[str, object]]) -> dict[str, object]:
    batch = [item for item in batch if item is not None]
    if not batch:
        raise RuntimeError("A full batch contained only unreadable images.")
    images = torch.stack([item["image"] for item in batch])
    targets = torch.stack([item["target"] for item in batch])
    return {
        "image": images,
        "target": targets,
        "path": [item["path"] for item in batch],
        "label": [item["label"] for item in batch],
    }
