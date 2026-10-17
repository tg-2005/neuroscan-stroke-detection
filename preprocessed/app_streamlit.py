"""Streamlit app for deployment-ready stroke CT inference."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from stroke_detection.constants import CLASS_NAMES
from stroke_detection.decision import DecisionConfig
from stroke_detection.inference import StrokePredictor


st.set_page_config(page_title="Stroke CT AI Assistant", layout="wide")


@st.cache_resource(show_spinner=False)
def load_predictor(
    googlenet_checkpoint: str,
    inception_checkpoint: str,
    googlenet_weight: float,
    inception_weight: float,
    device: str,
    decision_config_path: str,
    gradcam_resolution: str,
) -> StrokePredictor:
    decision_config = DecisionConfig.from_json(Path(decision_config_path)) if decision_config_path else None
    return StrokePredictor(
        googlenet_checkpoint=Path(googlenet_checkpoint) if googlenet_checkpoint else None,
        inception_v3_checkpoint=Path(inception_checkpoint) if inception_checkpoint else None,
        ensemble_weights=(googlenet_weight, inception_weight),
        device=device,
        decision_config=decision_config,
        gradcam_resolution=gradcam_resolution,
    )


st.title("AI-Assisted Stroke Detection from Preprocessed CT")
st.caption("Classes: 0=Bleeding, 1=Ischemia, 2=Normal. This is decision support, not a standalone diagnosis.")

with st.sidebar:
    st.header("Model Settings")
    googlenet_checkpoint = st.text_input(
        "GoogleNet checkpoint",
        "D:/preprocessed/outputs/googlenet/best_model.pt",
    )
    inception_checkpoint = st.text_input(
        "InceptionV3 checkpoint",
        "D:/preprocessed/outputs/inception_v3/best_model.pt",
    )
    googlenet_weight = st.slider("GoogleNet ensemble weight", 0.0, 1.0, 0.4, 0.05)
    inception_weight = st.slider("InceptionV3 ensemble weight", 0.0, 1.0, 0.6, 0.05)
    decision_config_path = st.text_input(
        "Decision threshold config",
        "D:/preprocessed/outputs/evaluation/ensemble/decision_thresholds.json",
    )
    gradcam_resolution = st.selectbox(
        "Grad-CAM spatial detail",
        ["balanced", "high", "final"],
        index=0,
        help="Balanced/high use earlier feature blocks for sharper maps; final is more semantic but blurrier.",
    )
    device = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    st.info("Inference uses the same 299x299 grayscale-to-RGB, mild CLAHE, ImageNet-normalized preprocessing as training.")

uploaded = st.file_uploader("Upload a preprocessed CT image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        image_path = Path(tmp.name)

    try:
        predictor = load_predictor(
            googlenet_checkpoint,
            inception_checkpoint,
            googlenet_weight,
            inception_weight,
            device,
            decision_config_path if Path(decision_config_path).exists() else "",
            gradcam_resolution,
        )
        with st.spinner("Running medically conservative inference and Grad-CAM..."):
            result = predictor.predict_ct(image_path, generate_explanations=True)

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.subheader("Prediction")
            st.metric("Predicted class", result.predicted_class)
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
            if result.predicted_class == "Normal" and (
                result.probabilities["Bleeding"] + result.probabilities["Ischemia"]
            ) > 0.35:
                st.warning("Stroke probability is non-trivial despite a Normal top class. Recommend careful clinical review.")
            if result.confidence < 0.60:
                st.warning("Low-confidence prediction. Treat as uncertain and review the scan manually.")

        with col_b:
            st.subheader("Class Probabilities")
            st.bar_chart({name: result.probabilities[name] for name in CLASS_NAMES})
            st.write(result.probabilities)

        st.subheader("Per-Model Probabilities")
        st.json(result.model_probabilities)

        st.subheader("Grad-CAM Explanations")
        for model_name, cam in result.gradcams.items():
            st.markdown(f"**{model_name}**")
            c1, c2, c3 = st.columns(3)
            c1.image(cam.original_rgb, caption="Original CT", use_container_width=True)
            c2.image(cam.heatmap_rgb, caption=f"Grad-CAM heatmap: {result.predicted_class}", use_container_width=True)
            c3.image(cam.overlay, caption=f"Overlay ({cam.target_resolution})", use_container_width=True)
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
else:
    st.write("Upload a CT image to run prediction and Grad-CAM visualization.")
