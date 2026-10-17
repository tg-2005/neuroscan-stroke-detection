from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import torch

from stroke_mri.inference import predict_mri


st.set_page_config(page_title="MRI Stroke Detection", layout="wide")
st.title("MRI Stroke Detection")

checkpoint = st.sidebar.text_input("Checkpoint path", value="outputs/run_resnet50/best_model.pt")
threshold = st.sidebar.slider("Stroke decision threshold", min_value=0.10, max_value=0.90, value=0.35, step=0.05)
uploaded = st.file_uploader("Upload MRI image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "dcm", "dicom"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    if not Path(checkpoint).exists():
        st.error(f"Checkpoint not found: {checkpoint}")
    else:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result = predict_mri(tmp_path, checkpoint, device=device, decision_threshold=threshold)
            st.subheader(f"{result['predicted_class']}  |  {result['confidence'] * 100:.1f}% confidence")

            cols = st.columns(3)
            cols[0].image(result["processed_image"], caption="Preprocessed MRI", use_container_width=True)
            cols[1].image(result["heatmap"], caption="Grad-CAM Heatmap", use_container_width=True)
            cols[2].image(result["overlay"], caption="Grad-CAM Overlay", use_container_width=True)

            st.write("Class probabilities")
            st.dataframe(
                [{"class": name, "probability": f"{prob * 100:.2f}%"} for name, prob in result["probabilities"].items()],
                hide_index=True,
                use_container_width=True,
            )
            st.caption("Clinical decision support only. Final diagnosis must be made by qualified clinicians with full imaging context.")
        except Exception as exc:
            st.error(f"Inference failed: {exc}")
