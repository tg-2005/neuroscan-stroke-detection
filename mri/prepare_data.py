from __future__ import annotations

import argparse
from pathlib import Path

from stroke_mri.dataset import create_grouped_manifests
from stroke_mri.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Create patient-aware MRI stroke manifests.")
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--class-mode", choices=["auto", "binary", "multiclass"], default="auto")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-readable-check", action="store_true")
    parser.add_argument(
        "--include-generated-augmented",
        action="store_true",
        help="Include files with generated augmentation prefixes such as std_aug_. Default is off to prevent augmented leakage.",
    )
    args = parser.parse_args()

    paths, spec = create_grouped_manifests(
        args.data_root,
        args.out_dir,
        class_mode=args.class_mode,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        validate_images=not args.skip_readable_check,
        include_generated_augmented=args.include_generated_augmented,
    )
    save_json({"class_names": spec.class_names, "class_to_idx": spec.class_to_idx, "mode": spec.mode}, args.out_dir / "class_mapping.json")
    for split, path in paths.items():
        print(f"{split}: {path}")
    print(f"class mapping: {spec.class_to_idx}")


if __name__ == "__main__":
    main()
