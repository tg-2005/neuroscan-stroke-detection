# MRI Stroke Detection with ResNet50 and Grad-CAM

Production-oriented PyTorch pipeline for MRI-based stroke classification with explainable Grad-CAM overlays.

## What Is Included

- Safe PNG/JPG/JPEG/DICOM loading with corrupted-image rejection.
- Grayscale-to-RGB conversion for ImageNet-pretrained ResNet50.
- Conservative MRI preprocessing: optional skull stripping, CLAHE, optional very mild denoising, 224 x 224 resizing, ImageNet normalization.
- Binary and multiclass support:
  - `Normal` vs `Stroke`
  - fixed multiclass mapping: `0 -> Ischemic`, `1 -> Hemorrhagic`, `2 -> Normal`, with extra dataset-specific classes appended after those.
- Patient-aware manifest splitting and leakage checks for patient overlap and exact duplicate hashes.
- Generated offline augmentation files such as `std_aug_*` are excluded by default during manifest creation to prevent augmented leakage.
- ResNet50 transfer learning with dropout, class imbalance handling, AdamW, ReduceLROnPlateau, early stopping, and checkpointing.
- Evaluation metrics: accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, confusion matrix, class-wise report.
- Class-specific Grad-CAM from ResNet50 `layer4[-1]` with heatmap and overlay generation. Binary `Normal` explanations use the negative stroke logit rather than falsely explaining stroke evidence.
- Streamlit deployment app.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Prepare Data

If your data is already in `dataset/train`, `dataset/val`, and `dataset/test`, you can train directly.

This workspace currently uses class folders at the root, so create leakage-aware manifests first:

```powershell
python prepare_data.py --data-root D:\mri --out-dir D:\mri\manifests --class-mode multiclass
```

For binary training that collapses every non-normal class into `Stroke`:

```powershell
python prepare_data.py --data-root D:\mri --out-dir D:\mri\manifests_binary --class-mode binary
```

Only use `--include-generated-augmented` if you have verified that generated files cannot leak across patients or splits.

## Train

```powershell
python train.py --config configs/default.yaml --data-root D:\mri --manifests-dir D:\mri\manifests
```

The best checkpoint is saved to:

```text
outputs/run_resnet50/best_model.pt
```

## Evaluate and Failure Analysis

```powershell
python evaluate.py --checkpoint outputs/run_resnet50/best_model.pt --manifest manifests/test.csv --out-dir outputs/evaluation
```

This writes metrics, predictions, confusion matrix, misclassified cases, low-confidence cases, and Grad-CAM panels.

## Streamlit Deployment

```powershell
streamlit run app_streamlit.py
```

Upload a MRI image and the app returns prediction, confidence, class probabilities, heatmap, and overlay. The app uses the exact preprocessing configuration stored in the training checkpoint.

## Inference from Python

```python
from stroke_mri.inference import predict_mri

result = predict_mri(
    image_path="example.png",
    checkpoint_path="outputs/run_resnet50/best_model.pt",
)
print(result["predicted_class"], result["confidence"])
```

## Inference from Command Line

```powershell
python -m stroke_mri.inference --image D:\mri\Normal\example.png --checkpoint outputs/run_resnet50/best_model.pt --overlay-out outputs\overlay.png
```

## Clinical Safety Notes

This software is clinical decision support, not a standalone diagnostic device. Validate on patient-level held-out data from the target scanner/site before clinical use. Tune the binary stroke decision threshold on validation data to prioritize sensitivity and reduce false negatives.
