from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json

import cv2
import numpy as np
import torch

from .gradcam import GradCAM, make_heatmap_overlay
from .model import build_model
from .preprocessing import PreprocessConfig, preprocess_for_inference


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    arch = checkpoint.get("model_arch", "resnet50")
    if arch == "hybrid_cbam_swin":
        from .hybrid_model import build_hybrid_model
        embed_dim = int(checkpoint.get("config", {}).get("embed_dim", 256))
        model = build_hybrid_model(
            num_classes=len(class_names),
            dropout=float(checkpoint.get("dropout", 0.4)),
            pretrained=False,
            embed_dim=embed_dim,
        )
    else:
        model = build_model(num_classes=len(class_names), dropout=float(checkpoint.get("dropout", 0.5)), pretrained=False)
    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint architecture mismatch for '{arch}'. Verify class_names, binary/multiclass mode, "
            "dropout, and classifier output dimensions match the saved checkpoint."
        ) from exc
    model.to(device)
    model.eval()
    return model, checkpoint



def logits_to_probabilities(logits: torch.Tensor, class_names: list[str]) -> torch.Tensor:
    if len(class_names) == 2 and logits.shape[1] == 1:
        stroke_prob = torch.sigmoid(logits[:, 0])
        if class_names == ["Normal", "Stroke"]:
            return torch.stack([1.0 - stroke_prob, stroke_prob], dim=1)
        return torch.stack([stroke_prob, 1.0 - stroke_prob], dim=1)
    return torch.softmax(logits, dim=1)


def predict_mri(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device: torch.device | None = None,
    decision_threshold: float | None = None,
    save_overlay_path: str | Path | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    preprocess_config = PreprocessConfig(**checkpoint["preprocess_config"])
    class_names = checkpoint["class_names"]
    threshold = float(decision_threshold if decision_threshold is not None else checkpoint.get("decision_threshold", 0.5))

    tensor, processed = preprocess_for_inference(image_path, preprocess_config)
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = logits_to_probabilities(logits, class_names)[0].detach().cpu().numpy()

    if class_names == ["Normal", "Stroke"] and threshold < 0.5:
        stroke_index = class_names.index("Stroke")
        pred_idx = stroke_index if probabilities[stroke_index] >= threshold else class_names.index("Normal")
    else:
        pred_idx = int(np.argmax(probabilities))

    cam_engine = GradCAM(model, model.gradcam_target_layer, class_names=class_names)
    try:
        cam = cam_engine(tensor, target_index=pred_idx)
    finally:
        cam_engine.remove()
    heatmap, overlay = make_heatmap_overlay(processed, cam)

    if save_overlay_path is not None:
        out = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        Path(save_overlay_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_overlay_path), out)

    return {
        "predicted_class": class_names[pred_idx],
        "confidence": float(probabilities[pred_idx]),
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(class_names)},
        "processed_image": processed,
        "heatmap": heatmap,
        "overlay": overlay,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MRI stroke inference with Grad-CAM.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--overlay-out", type=Path, default=None)
    parser.add_argument("--decision-threshold", type=float, default=None)
    args = parser.parse_args()

    result = predict_mri(
        args.image,
        args.checkpoint,
        decision_threshold=args.decision_threshold,
        save_overlay_path=args.overlay_out,
    )
    serializable = {
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "overlay_path": str(args.overlay_out) if args.overlay_out else None,
    }
    print(json.dumps(serializable, indent=2))


if __name__ == "__main__":
    main()
