"""Configuration objects for reproducible training and deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class TrainConfig:
    data_root: Path = Path("D:/preprocessed")
    output_dir: Path = Path("D:/preprocessed/outputs")
    model_name: str = "inception_v3"
    image_size: int = 299
    batch_size: int = 16
    num_epochs: int = 40
    freeze_epochs: int = 5
    full_finetune_epoch: int = 18
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-7
    weight_decay: float = 1e-4
    dropout: float = 0.5
    num_workers: int = 2
    seed: int = 42
    patience: int = 8
    use_weighted_sampler: bool = False
    use_class_weights: bool = True
    ischemia_loss_boost: float = 1.10
    device: str = "auto"
    pretrained: bool = True
    max_grad_norm: float = 5.0
    gradcam_every_epoch: bool = False
    loss_name: str = "focal"
    focal_gamma: float = 1.5
    label_smoothing: float = 0.03

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_dir"] = str(self.output_dir)
        return payload


@dataclass
class EnsembleConfig:
    googlenet_checkpoint: Path = Path("D:/preprocessed/outputs/googlenet/best_model.pt")
    inception_v3_checkpoint: Path = Path("D:/preprocessed/outputs/inception_v3/best_model.pt")
    weights: Tuple[float, float] = (0.4, 0.6)
    device: str = "auto"
