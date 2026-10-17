from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from stroke_mri.dataset import MRIDataset, _canonical_label, collate_skip_bad
from stroke_mri.gradcam import gradcam_for_path, save_gradcam_panel
from stroke_mri.inference import load_checkpoint, logits_to_probabilities
from stroke_mri.metrics import compute_metrics, save_confusion_matrix
from stroke_mri.preprocessing import PreprocessConfig
from stroke_mri.utils import ensure_dir, get_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MRI stroke model and generate failure-analysis Grad-CAMs.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-failures", type=int, default=40)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    device = get_device()
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    class_names = checkpoint["class_names"]
    decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
    preprocess_cfg = PreprocessConfig(**checkpoint["preprocess_config"])

    df = pd.read_csv(args.manifest)
    label_column = "label" if "label" in df.columns else "source_label"
    if label_column in df.columns:
        mode = "binary" if class_names == ["Normal", "Stroke"] else "multiclass"
        class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        df = df.copy()
        df["label"] = df[label_column].map(lambda x: _canonical_label(str(x), mode))
        df["target"] = df["label"].map(class_to_idx)
        if df["target"].isna().any():
            missing = sorted(df.loc[df["target"].isna(), "label"].astype(str).unique())
            raise ValueError(f"Manifest contains labels not present in checkpoint class mapping: {missing}")
        df["target"] = df["target"].astype(int)
    dataset = MRIDataset(df, preprocess_cfg, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_skip_bad)

    y_true, y_pred, y_score, rows = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = model(images)
            probs = logits_to_probabilities(logits, class_names).cpu().numpy()
            if class_names == ["Normal", "Stroke"]:
                stroke_index = class_names.index("Stroke")
                normal_index = class_names.index("Normal")
                preds = np.where(probs[:, stroke_index] >= decision_threshold, stroke_index, normal_index)
            else:
                preds = probs.argmax(axis=1)
            y_true.extend(targets.cpu().numpy().tolist())
            y_pred.extend(preds.tolist())
            y_score.append(probs)
            for i, path in enumerate(batch["path"]):
                rows.append(
                    {
                        "path": path,
                        "actual": class_names[int(targets[i].cpu())],
                        "predicted": class_names[int(preds[i])],
                        "confidence": float(probs[i, preds[i]]),
                        "correct": bool(int(targets[i].cpu()) == int(preds[i])),
                    }
                )

    scores = np.concatenate(y_score, axis=0)
    metrics = compute_metrics(y_true, y_pred, scores, class_names)
    save_json(metrics, out_dir / "metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], class_names, out_dir / "confusion_matrix.png")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_dir / "predictions.csv", index=False)
    failures = result_df[~result_df["correct"]].sort_values("confidence", ascending=False).head(args.max_failures)
    low_conf = result_df[result_df["confidence"] < 0.60].sort_values("confidence").head(args.max_failures)

    for name, subset in {"misclassified": failures, "low_confidence": low_conf}.items():
        folder = ensure_dir(out_dir / name)
        for idx, row in subset.iterrows():
            pred_idx = class_names.index(row["predicted"])
            original, heatmap, overlay = gradcam_for_path(
                model,
                model.gradcam_target_layer,
                row["path"],
                preprocess_cfg,
                device,
                pred_idx,
                class_names=class_names,
            )
            safe_name = f"{idx}_{row['actual']}_as_{row['predicted']}.png".replace("/", "_").replace("\\", "_")
            save_gradcam_panel(original, heatmap, overlay, folder / safe_name, title=f"{row['actual']} -> {row['predicted']} ({row['confidence']:.2f})")

    print(f"Saved metrics and failure analysis to {out_dir}")


if __name__ == "__main__":
    main()
