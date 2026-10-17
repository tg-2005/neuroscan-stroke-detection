# Stroke CT Detection System

Production-oriented PyTorch pipeline for 3-class stroke CT classification:

- `0 -> Bleeding`
- `1 -> Ischemia`
- `2 -> Normal`

The implementation uses GoogleNet and InceptionV3 with ImageNet transfer learning, conservative CT preprocessing, class-imbalance handling, Grad-CAM explainability, soft-voting ensemble inference, and Streamlit deployment.

## Install

```powershell
cd D:\preprocessed
python -m pip install -r requirements.txt
```

## Dataset Layout

The code supports both:

```text
D:/preprocessed/train/Bleeding
D:/preprocessed/val/Ischemia
D:/preprocessed/test/Normal
```

and your current extracted layout:

```text
D:/preprocessed/train-.../train/Bleeding
D:/preprocessed/val-.../val/Ischemia
D:/preprocessed/test-.../test/Normal
```

## Audit First

```powershell
python -m stroke_detection.audit_dataset --data-root D:\preprocessed
```

This checks class counts, corrupted images, and likely patient/image overlap across `train`, `val`, and `test`. If overlap is detected, training stops because leakage can produce clinically misleading performance.

## Train Both Models

```powershell
python -m stroke_detection.train --data-root D:\preprocessed --model both --epochs 40 --batch-size 16
```

Outputs are saved to:

```text
D:/preprocessed/outputs/googlenet/best_model.pt
D:/preprocessed/outputs/inception_v3/best_model.pt
```

Checkpoint selection prioritizes ischemia recall and overall stroke sensitivity, while also requiring Normal recall / stroke specificity so an all-stroke classifier cannot be selected as the best model.

## Evaluate

```powershell
python -m stroke_detection.evaluate --model googlenet --checkpoint D:\preprocessed\outputs\googlenet\best_model.pt
python -m stroke_detection.evaluate --model inception_v3 --checkpoint D:\preprocessed\outputs\inception_v3\best_model.pt
python -m stroke_detection.evaluate_ensemble
```

## Tune For Higher Ischemia Sensitivity

If accuracy is good but ischemia recall is too low, tune the operating point on the validation split:

```powershell
python -m stroke_detection.tune_decision_thresholds --data-root D:\preprocessed --googlenet-checkpoint D:\preprocessed\outputs\googlenet\best_model.pt --inception-checkpoint D:\preprocessed\outputs\inception_v3\best_model.pt --min-ischemia-recall 0.85 --min-stroke-sensitivity 0.90 --output D:\preprocessed\outputs\evaluation\ensemble\decision_thresholds.json

python -m stroke_detection.evaluate_ensemble --decision-config D:\preprocessed\outputs\evaluation\ensemble\decision_thresholds.json --output-dir D:\preprocessed\outputs\evaluation\ensemble_tuned
```

This does not retrain the model. It adjusts class decision multipliers learned from validation data and is useful when you want fewer missed ischemia cases.

If you want to evaluate the final epoch checkpoint instead of the selected best checkpoint:

```powershell
python -m stroke_detection.evaluate --model googlenet --checkpoint D:\preprocessed\outputs\googlenet\last_model.pt
python -m stroke_detection.evaluate --model inception_v3 --checkpoint D:\preprocessed\outputs\inception_v3\last_model.pt
python -m stroke_detection.evaluate_ensemble --googlenet-checkpoint D:\preprocessed\outputs\googlenet\last_model.pt --inception-checkpoint D:\preprocessed\outputs\inception_v3\last_model.pt
```

Evaluation exports metrics, confusion matrices, classification reports, failure analysis CSVs, and Grad-CAM images for selected mistakes.

## Inference From Python

```python
from stroke_detection.inference import predict_ct

result = predict_ct("D:/preprocessed/test-20260405T202619Z-1-001/test/Ischemia/example.png")
print(result.predicted_class, result.confidence, result.probabilities)
```

## Streamlit Deployment

```powershell
streamlit run D:\preprocessed\app_streamlit.py
```

The app supports CPU/GPU inference, image upload, ensemble confidence scores, per-model probabilities, and Grad-CAM overlays.

Use the `Decision threshold config` field in the sidebar to load:

```text
D:/preprocessed/outputs/evaluation/ensemble/decision_thresholds.json
```

Use `Grad-CAM spatial detail = balanced` or `high` for less blocky heatmaps. `final` is more semantic but blurrier because Inception/GoogleNet final feature maps are very coarse.

## Medical Reliability Notes

- Inference preprocessing exactly matches training preprocessing.
- Images are loaded as grayscale, mildly CLAHE-enhanced, converted with `cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)`, resized to `299x299`, and ImageNet-normalized.
- No aggressive blur is used. Very light denoising is disabled by default to preserve subtle ischemic texture changes.
- Training uses class-weighted loss with a modest ischemia boost to reduce dangerous false negatives without collapsing Normal specificity. Weighted sampling is available via `--use-weighted-sampler`, but do not combine aggressive sampling and strong class weights unless validation specificity remains acceptable.
- This system is clinical decision support only and must be validated externally before any real-world medical use.
