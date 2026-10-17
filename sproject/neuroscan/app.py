"""
NeuroScan Flask backend.

This app serves the frontend in index.html and adapts it to the real model
projects on the D drive.

MRI backend: D:\mri
CT backend : D:\preprocessed
"""

from __future__ import annotations

import io
import json
import os
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS


BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "uploads"
GRADCAM_DIR = BASE / "static" / "gradcam"
ORIGINAL_DIR = BASE / "static" / "originals"
STATS_FILE = BASE / "stats.json"

PROJECT_ROOT = BASE.parents[1] if len(BASE.parents) > 1 else BASE.parent
DEFAULT_MRI = PROJECT_ROOT / "mri" if (PROJECT_ROOT / "mri").exists() else Path(r"D:\mri")
DEFAULT_CT = PROJECT_ROOT / "preprocessed" if (PROJECT_ROOT / "preprocessed").exists() else Path(r"D:\preprocessed")

MRI_ROOT = Path(os.getenv("NEUROSCAN_MRI_ROOT", str(DEFAULT_MRI)))
CT_ROOT = Path(os.getenv("NEUROSCAN_CT_ROOT", str(DEFAULT_CT)))

HYBRID_MRI_CHECKPOINT = MRI_ROOT / "outputs" / "run_hybrid_cbam_swin" / "best_model.pt"
RESNET_MRI_CHECKPOINT = MRI_ROOT / "outputs" / "run_resnet50" / "best_model.pt"
DEFAULT_MRI_CHECKPOINT = HYBRID_MRI_CHECKPOINT if HYBRID_MRI_CHECKPOINT.exists() else RESNET_MRI_CHECKPOINT

MRI_CHECKPOINT = Path(
    os.getenv("NEUROSCAN_MRI_CHECKPOINT", str(DEFAULT_MRI_CHECKPOINT))
)
CT_GOOGLENET_CHECKPOINT = Path(
    os.getenv("NEUROSCAN_CT_GOOGLENET_CHECKPOINT", str(CT_ROOT / "outputs" / "googlenet" / "best_model.pt"))
)
CT_INCEPTION_CHECKPOINT = Path(
    os.getenv("NEUROSCAN_CT_INCEPTION_CHECKPOINT", str(CT_ROOT / "outputs" / "inception_v3" / "best_model.pt"))
)
CT_HYBRID_CHECKPOINT = Path(
    os.getenv("NEUROSCAN_CT_HYBRID_CHECKPOINT", str(CT_ROOT / "outputs" / "hybrid_cbam_swin" / "best_model.pt"))
)
CT_DECISION_CONFIG = Path(
    os.getenv(
        "NEUROSCAN_CT_DECISION_CONFIG",
        str(CT_ROOT / "outputs" / "evaluation" / "ensemble" / "decision_thresholds.json"),
    )
)

for directory in (UPLOAD_DIR, GRADCAM_DIR, ORIGINAL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for project_root in (str(MRI_ROOT), str(CT_ROOT)):
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


try:
    import torch

    from stroke_detection.decision import DecisionConfig
    from stroke_detection.inference import StrokePredictor
    from stroke_mri.gradcam import GradCAM as MRIGradCAM
    from stroke_mri.gradcam import make_heatmap_overlay as make_mri_heatmap_overlay
    from stroke_mri.inference import load_checkpoint as load_mri_checkpoint
    from stroke_mri.inference import logits_to_probabilities as mri_logits_to_probabilities
    from stroke_mri.preprocessing import PreprocessConfig as MRIPreprocessConfig
    from stroke_mri.preprocessing import preprocess_for_inference as preprocess_mri_for_inference

    MODEL_DEPS_OK = True
    MODEL_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - surfaced through /api/models
    torch = None  # type: ignore[assignment]
    DecisionConfig = None  # type: ignore[assignment]
    StrokePredictor = None  # type: ignore[assignment]
    MRIGradCAM = None  # type: ignore[assignment]
    make_mri_heatmap_overlay = None  # type: ignore[assignment]
    load_mri_checkpoint = None  # type: ignore[assignment]
    mri_logits_to_probabilities = None  # type: ignore[assignment]
    MRIPreprocessConfig = None  # type: ignore[assignment]
    preprocess_mri_for_inference = None  # type: ignore[assignment]
    MODEL_DEPS_OK = False
    MODEL_IMPORT_ERROR = str(exc)


app = Flask(__name__, static_folder="static")
CORS(app)


ct_predictor: Any | None = None
ct_error = ""
CT_LOADED = False
ct_lock = threading.Lock()

mri_model: Any | None = None
mri_checkpoint: dict[str, Any] | None = None
mri_preprocess_config: Any | None = None
mri_class_names: list[str] = []
mri_decision_threshold = 0.5
mri_device: Any | None = None
mri_error = ""
MRI_LOADED = False
mri_lock = threading.Lock()


def _load_stats() -> dict[str, int]:
    try:
        stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        return {"total": int(stats.get("total", 0))}
    except Exception:
        return {"total": 0}


def _save_stats(stats: dict[str, int]) -> None:
    STATS_FILE.write_text(json.dumps(stats), encoding="utf-8")


def _increment_stats() -> int:
    stats = _load_stats()
    stats["total"] += 1
    _save_stats(stats)
    return stats["total"]


def _json_error(message: str, status: int = 400):
    return jsonify({"success": False, "error": message}), status


def _device_string() -> str:
    return os.getenv("NEUROSCAN_DEVICE", "auto").strip().lower() or "auto"


def _torch_device() -> Any:
    requested = _device_string()
    if requested == "cuda" and torch is not None and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch is not None and torch.cuda.is_available() else "cpu")


def load_ct_model() -> bool:
    global CT_LOADED, ct_error, ct_predictor
    ct_error = ""
    CT_LOADED = False

    if not MODEL_DEPS_OK:
        ct_error = f"Model dependencies failed to import: {MODEL_IMPORT_ERROR}"
        return False

    if not CT_GOOGLENET_CHECKPOINT.exists() and not CT_INCEPTION_CHECKPOINT.exists() and not CT_HYBRID_CHECKPOINT.exists():
        ct_error = "CT checkpoints not found under D:\\preprocessed\\outputs."
        return False

    try:
        decision_config = DecisionConfig.from_json(CT_DECISION_CONFIG) if CT_DECISION_CONFIG.exists() else None
        ct_predictor = StrokePredictor(
            googlenet_checkpoint=CT_GOOGLENET_CHECKPOINT if CT_GOOGLENET_CHECKPOINT.exists() else None,
            inception_v3_checkpoint=CT_INCEPTION_CHECKPOINT if CT_INCEPTION_CHECKPOINT.exists() else None,
            hybrid_checkpoint=CT_HYBRID_CHECKPOINT if CT_HYBRID_CHECKPOINT.exists() else None,
            ensemble_weights=(0.3, 0.35, 0.35),
            device=_device_string(),
            decision_config=decision_config,
            gradcam_resolution=os.getenv("NEUROSCAN_CT_GRADCAM", "balanced"),
        )
        CT_LOADED = True
        return True
    except Exception as exc:
        ct_error = str(exc)
        ct_predictor = None
        return False


def load_mri_model() -> bool:
    global MRI_LOADED, mri_error, mri_model, mri_checkpoint
    global mri_preprocess_config, mri_class_names, mri_decision_threshold, mri_device

    mri_error = ""
    MRI_LOADED = False

    if not MODEL_DEPS_OK:
        mri_error = f"Model dependencies failed to import: {MODEL_IMPORT_ERROR}"
        return False

    if not MRI_CHECKPOINT.exists():
        mri_error = f"MRI checkpoint not found: {MRI_CHECKPOINT}"
        return False

    try:
        mri_device = _torch_device()
        mri_model, mri_checkpoint = load_mri_checkpoint(MRI_CHECKPOINT, mri_device)
        mri_preprocess_config = MRIPreprocessConfig(**mri_checkpoint["preprocess_config"])
        mri_class_names = list(mri_checkpoint["class_names"])
        mri_decision_threshold = float(mri_checkpoint.get("decision_threshold", 0.5))
        MRI_LOADED = True
        return True
    except Exception as exc:
        mri_error = str(exc)
        mri_model = None
        mri_checkpoint = None
        return False


def _normalize_display_class(raw_class: str) -> str:
    value = raw_class.replace("_", " ").strip()
    lowered = value.lower()
    if "normal" in lowered:
        return "Normal"
    if "ische" in lowered:
        return "Ischemic Stroke"
    if "hemorr" in lowered or "bleed" in lowered:
        return "Hemorrhagic Stroke"
    if "stroke" in lowered:
        return "Stroke - Subtype Uncertain"
    return value.title()


def _display_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for raw_name, probability in probabilities.items():
        display_name = _normalize_display_class(raw_name)
        normalized[display_name] = normalized.get(display_name, 0.0) + float(probability)
    return {name: round(value * 100.0, 2) for name, value in normalized.items()}


def _stroke_detected(display_class: str) -> bool:
    return display_class != "Normal"


def _severity_for(display_class: str) -> str:
    if display_class == "Normal":
        return "None"
    if display_class == "Hemorrhagic Stroke":
        return "Severe"
    return "Moderate-Severe"


def _region_for(display_class: str) -> str:
    if display_class == "Normal":
        return "N/A"
    if display_class == "Ischemic Stroke":
        return "Middle Cerebral Artery Territory"
    if display_class == "Hemorrhagic Stroke":
        return "Basal Ganglia / Cortex"
    return "Stroke territory not subtype-classified"


def _recommendations_for(display_class: str) -> list[str]:
    recommendations = {
        "Normal": [
            "No acute stroke finding from the selected model.",
            "Correlate with clinical presentation and scanner context.",
            "Schedule follow-up as clinically indicated.",
        ],
        "Ischemic Stroke": [
            "Activate stroke team and urgent neurology consultation.",
            "Assess IV thrombolysis eligibility within the treatment window.",
            "Consider CT/MR angiography to evaluate vessel occlusion.",
            "Monitor BP and glucose per acute stroke protocol.",
            "NPO pending swallowing assessment.",
        ],
        "Hemorrhagic Stroke": [
            "Immediate neurosurgery or neurocritical care consultation.",
            "Review anticoagulant or antiplatelet use urgently.",
            "Control blood pressure per hemorrhage protocol.",
            "ICU-level monitoring and repeat imaging as clinically indicated.",
            "Assess for expansion, mass effect, and raised ICP.",
        ],
    }
    return recommendations.get(
        display_class,
        [
            "Stroke probability is elevated, but subtype is uncertain.",
            "Request urgent radiologist or neurologist review.",
            "Use additional imaging and clinical findings before treatment decisions.",
        ],
    )


def _safe_percent(value: float) -> float:
    return round(float(value) * 100.0, 1)


def _save_rgb_png(image_rgb: np.ndarray, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.asarray(image_rgb)
    if image.ndim == 2:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)
    return f"/static/{output_path.parent.name}/{output_path.name}"


def _save_upload(file_storage: Any, scan_id: str, modality: str) -> Path:
    filename = file_storage.filename or ""
    suffix = Path(filename).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm", ".dicom"}:
        suffix = ".png"
    path = UPLOAD_DIR / f"{scan_id}_{modality}{suffix}"
    file_storage.save(path)
    return path


def _dicom_to_png(input_path: Path, output_path: Path) -> Path:
    try:
        import pydicom
    except Exception as exc:
        raise RuntimeError("DICOM CT uploads require pydicom to be installed.") from exc

    ds = pydicom.dcmread(str(input_path))
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
    arr = arr - np.min(arr)
    arr = arr / (np.max(arr) + 1e-8)
    image = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(output_path)
    return output_path


def _ct_image_path_for_inference(upload_path: Path) -> Path:
    if upload_path.suffix.lower() not in {".dcm", ".dicom"}:
        return upload_path
    return _dicom_to_png(upload_path, upload_path.with_suffix(".png"))


def _predict_ct(upload_path: Path, scan_id: str) -> dict[str, Any]:
    if not CT_LOADED or ct_predictor is None:
        raise RuntimeError(f"CT model is not loaded. {ct_error}".strip())

    image_path = _ct_image_path_for_inference(upload_path)
    with ct_lock:
        result = ct_predictor.predict_ct(image_path, generate_explanations=True)

    display_class = _normalize_display_class(result.predicted_class)
    selected_gradcam = None
    if result.gradcams:
        selected_gradcam = (
            result.gradcams.get("hybrid_cbam_swin")
            or result.gradcams.get("inception_v3")
            or next(iter(result.gradcams.values()))
        )

    original_url = None
    heatmap_url = None
    if selected_gradcam is not None:
        original_url = _save_rgb_png(selected_gradcam.original_rgb, ORIGINAL_DIR / f"{scan_id}_ct_original.png")
        heatmap_url = _save_rgb_png(selected_gradcam.overlay, GRADCAM_DIR / f"{scan_id}_ct_overlay.png")

    active_names = list(ct_predictor.models.keys()) if ct_predictor else ["googlenet", "inception_v3"]
    model_display = "CT Ensemble: " + " + ".join(
        "Hybrid CNN-ViT" if n == "hybrid_cbam_swin" else ("InceptionV3" if n == "inception_v3" else "GoogLeNet")
        for n in active_names
    )

    return {
        "display_class": display_class,
        "confidence": _safe_percent(result.confidence),
        "class_probabilities": _display_probabilities(result.probabilities),
        "raw_class_probabilities": {name: round(value * 100.0, 2) for name, value in result.probabilities.items()},
        "model_probabilities": {
            model_name: {name: round(value * 100.0, 2) for name, value in probabilities.items()}
            for model_name, probabilities in result.model_probabilities.items()
        },
        "model_used": model_display,
        "original_url": original_url,
        "heatmap_url": heatmap_url,
    }


def _predict_mri(upload_path: Path, scan_id: str) -> dict[str, Any]:
    if not MRI_LOADED or mri_model is None:
        raise RuntimeError(f"MRI model is not loaded. {mri_error}".strip())

    with mri_lock:
        tensor, processed = preprocess_mri_for_inference(upload_path, mri_preprocess_config)
        tensor = tensor.unsqueeze(0).to(mri_device)
        with torch.no_grad():
            logits = mri_model(tensor)
            probabilities = mri_logits_to_probabilities(logits, mri_class_names)[0].detach().cpu().numpy()

        if mri_class_names == ["Normal", "Stroke"] and mri_decision_threshold < 0.5:
            stroke_index = mri_class_names.index("Stroke")
            pred_idx = stroke_index if probabilities[stroke_index] >= mri_decision_threshold else mri_class_names.index("Normal")
        else:
            pred_idx = int(np.argmax(probabilities))

        cam_engine = MRIGradCAM(mri_model, mri_model.gradcam_target_layer, class_names=mri_class_names)
        try:
            cam = cam_engine(tensor, target_index=pred_idx)
        finally:
            cam_engine.remove()
        _heatmap, overlay = make_mri_heatmap_overlay(processed, cam)

    probabilities_by_name = {name: float(probabilities[index]) for index, name in enumerate(mri_class_names)}
    display_class = _normalize_display_class(mri_class_names[pred_idx])

    original_url = _save_rgb_png(processed, ORIGINAL_DIR / f"{scan_id}_mri_original.png")
    heatmap_url = _save_rgb_png(overlay, GRADCAM_DIR / f"{scan_id}_mri_overlay.png")

    arch_name = mri_checkpoint.get("model_arch", "ResNet-50") if mri_checkpoint else "ResNet-50"
    model_display = "MRI Hybrid CNN-Transformer (ResNet50+CBAM + Swin-T)" if arch_name == "hybrid_cbam_swin" else f"MRI {arch_name.title()}"

    return {
        "display_class": display_class,
        "confidence": _safe_percent(float(probabilities[pred_idx])),
        "class_probabilities": _display_probabilities(probabilities_by_name),
        "raw_class_probabilities": {name: round(value * 100.0, 2) for name, value in probabilities_by_name.items()},
        "model_used": model_display,
        "original_url": original_url,
        "heatmap_url": heatmap_url,
    }


@app.route("/")
def index():
    return send_file(BASE / "index.html")


@app.route("/api/models")
def api_models():
    return jsonify(
        {
            "success": True,
            "data": {
                "ct": {
                    "loaded": CT_LOADED,
                    "name": (
                        "GoogLeNet + InceptionV3 + Hybrid CNN-ViT Ensemble"
                        if (ct_predictor and "hybrid_cbam_swin" in ct_predictor.models)
                        else "GoogLeNet + InceptionV3 Ensemble"
                    ),
                    "active_models": list(ct_predictor.models.keys()) if ct_predictor else [],
                    "root": str(CT_ROOT),
                    "checkpoints": {
                        "googlenet": str(CT_GOOGLENET_CHECKPOINT),
                        "inception_v3": str(CT_INCEPTION_CHECKPOINT),
                        "hybrid_cbam_swin": str(CT_HYBRID_CHECKPOINT),
                    },
                    "decision_config": str(CT_DECISION_CONFIG) if CT_DECISION_CONFIG.exists() else None,
                    "error": ct_error,
                },
                "mri": {
                    "loaded": MRI_LOADED,
                    "name": (
                        "Hybrid CNN-Transformer (ResNet50+CBAM + Swin-T)"
                        if (mri_checkpoint and mri_checkpoint.get("model_arch") == "hybrid_cbam_swin")
                        else "ResNet-50 MRI classifier"
                    ),
                    "model_arch": (mri_checkpoint.get("model_arch", "resnet50") if mri_checkpoint else "resnet50"),
                    "root": str(MRI_ROOT),
                    "checkpoint": str(MRI_CHECKPOINT),
                    "classes": mri_class_names,
                    "decision_threshold": mri_decision_threshold,
                    "error": mri_error,
                },
            },
        }
    )


@app.route("/api/stats")
def api_stats():
    return jsonify({"success": True, "data": _load_stats()})


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        uploaded = request.files.get("file")
        modality = (request.form.get("modality") or "ct").strip().lower()
        patient_name = request.form.get("patient_name") or "Anonymous"
        patient_id = request.form.get("patient_id") or f"NS-{uuid.uuid4().hex[:6].upper()}"

        if uploaded is None:
            return _json_error("No file uploaded.", 400)
        if modality not in {"ct", "mri"}:
            return _json_error("Modality must be either 'ct' or 'mri'.", 400)

        scan_id = uuid.uuid4().hex
        upload_path = _save_upload(uploaded, scan_id, modality)

        if modality == "ct":
            prediction = _predict_ct(upload_path, scan_id)
        else:
            prediction = _predict_mri(upload_path, scan_id)

        display_class = prediction["display_class"]
        _increment_stats()

        payload = {
            "scan_id": scan_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "modality": modality.upper(),
            "patient_name": patient_name,
            "patient_id": patient_id,
            "stroke_detected": _stroke_detected(display_class),
            "stroke_type": display_class,
            "confidence": prediction["confidence"],
            "severity": _severity_for(display_class),
            "affected_region": _region_for(display_class),
            "model_used": prediction["model_used"],
            "is_demo": False,
            "recommendation": _recommendations_for(display_class),
            "class_probabilities": prediction["class_probabilities"],
            "raw_class_probabilities": prediction["raw_class_probabilities"],
            "heatmap_url": prediction["heatmap_url"],
            "original_url": prediction["original_url"],
        }
        if "model_probabilities" in prediction:
            payload["model_probabilities"] = prediction["model_probabilities"]

        return jsonify({"success": True, "data": payload})
    except Exception as exc:
        traceback.print_exc()
        return _json_error(str(exc), 500)


@app.route("/static/gradcam/<path:name>")
def serve_gradcam(name: str):
    return send_from_directory(GRADCAM_DIR, name)


@app.route("/static/originals/<path:name>")
def serve_original(name: str):
    return send_from_directory(ORIGINAL_DIR, name)


@app.route("/api/health")
def api_health():
    return jsonify({"success": True, "data": {"ct_loaded": CT_LOADED, "mri_loaded": MRI_LOADED}})


if __name__ == "__main__":
    print("Loading NeuroScan models...")
    print(f"  MRI root: {MRI_ROOT}")
    print(f"  CT root : {CT_ROOT}")
    load_ct_model()
    load_mri_model()
    print(f"  CT loaded : {CT_LOADED} {ct_error}")
    print(f"  MRI loaded: {MRI_LOADED} {mri_error}")
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
