# NeuroScan: Early Stroke Detection Using Deep Learning & Neuroimage Interpretation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Flask Backend](https://img.shields.io/badge/backend-Flask-green.svg)](https://flask.palletsprojects.com/)
[![License: Academic/Research](https://img.shields.io/badge/license-Academic%20Research-blue.svg)](#disclaimer)

**NeuroScan** is an advanced, dual-modality clinical decision support system designed for rapid, automated stroke detection and lesion localization from non-contrast Head CT and Brain MRI neuroimages.

---

## 🌟 Key Architecture & Highlights

### 1. ⬛ Non-Contrast CT Scan: 3-Model Soft-Voting Ensemble
- **Components**:
  - **GoogLeNet** (Inception-v1, Weight: 30%)
  - **InceptionV3** (Multi-scale asymmetric factorized convolutions, Weight: 35%)
  - **Hybrid CNN-ViT** (ResNet50 + CBAM Attention + Swin-Transformer, Weight: 35%)
- **Trained on**: 6,650 CT scans with Class-Weighted Focal Loss ($\gamma=2.0$) and Weighted Sampler to solve severe (66%) Normal class imbalance.
- **Performance**:
  - **ROC-AUC (Macro OVR)**: **96.70%**
  - **Ischemic Stroke Recall**: **90.00%** *(vs. 82.9% baseline)*
  - **Stroke Sensitivity**: **93.11%** *(detects 311 of 334 acute strokes on test set)*

### 2. 🔴 Brain MRI Scan: Hybrid CNN-Transformer (CAFM)
- **Components**:
  - **Local Branch**: ResNet-50 backbone with **CBAM** (Convolutional Block Attention Module) at Layers 3 & 4 for focal lesion capture.
  - **Global Branch**: Pretrained **Swin-Transformer (Swin-T)** for global brain hemispheric context.
  - **Fusion**: **CAFM** (Cross-Attention Feature Fusion Module) query-key-value token alignment.
- **Performance**:
  - **ROC-AUC (Macro OVR)**: **99.94%**
  - **Overall Accuracy**: **97.25%**
  - **Ischemic Sensitivity**: **97.83%**
  - **Hemorrhagic Sensitivity**: **95.74%**

### 3. 🗺️ Explainable AI (Grad-CAM)
- Native Grad-CAM visual heatmaps generate overlaid activation zones pinpointing the acute ischemic territory or hematoma.
- Automatic background skull/air filtering ensures zero artifact halos.

### 4. 📄 Branded Clinical PDF Export
- Instant 1-click export generating hospital-ready documentation:
  - Patient metadata & scan identification
  - Original scan slice and Grad-CAM overlay side-by-side
  - Detailed class probability distribution table
  - Emergency clinical triage guidelines & legal disclaimer

---

## 📁 Directory Structure

```
D:\final\
├── run_neuroscan.bat            # 🚀 ONE-CLICK LAUNCHER for Windows
├── README.md                    # Project documentation
│
├── sproject\neuroscan\          # 🌐 Web UI & API Application
│   ├── app.py                   # Flask server & inference adapter
│   ├── index.html               # Responsive frontend dashboard with PDF export
│   └── static\                  # Generated Grad-CAM overlays & original slices
│
├── preprocessed\                # ⬛ CT Pipeline (6,650 scans)
│   ├── stroke_detection\        # CT model architectures, dataset, inference engine
│   └── outputs\                 # Trained checkpoints (GoogLeNet, InceptionV3, Hybrid)
│       └── hybrid_cbam_swin\best_model.pt
│
└── mri\                         # 🔴 MRI Pipeline (5,000+ scans)
    ├── stroke_mri\              # CBAM, Swin-T, CAFM fusion, Grad-CAM modules
    └── outputs\                 # Trained checkpoint (ResNet50+CBAM + Swin-T)
        └── run_hybrid_cbam_swin\best_model.pt
```

---

## 🚀 How to Run the Project

### Method 1: One-Click (Recommended)
Simply double-click:
```bat
D:\final\run_neuroscan.bat
```
The script will:
1. Verify Python 3.11 environment.
2. Confirm CT 3-Model and MRI Hybrid checkpoints are **[READY]**.
3. Launch your default browser to **`http://localhost:5000`**.
4. Start the NeuroScan backend server.

### Method 2: Manual Terminal
```powershell
cd D:\final\sproject\neuroscan
python app.py
```
Open **`http://localhost:5000`** in any web browser.

---

## 🧪 Testing with Real Clinical Scans

You can test the system immediately with pre-loaded scans:

| Modality | Class | Test File Path |
|:---|:---|:---|
| **CT** | **Hemorrhagic** | `D:\final\preprocessed\test-20260405T202619Z-1-001\test\Bleeding\10039.png` |
| **CT** | **Ischemic** | `D:\final\preprocessed\test-20260405T202619Z-1-001\test\Ischemia\10024.png` |
| **CT** | **Normal** | `D:\final\preprocessed\test-20260405T202619Z-1-001\test\Normal\10010.png` |
| **MRI** | **Ischemic** | `D:\final\mri\Ischemic\ocmri_350.png` |
| **MRI** | **Hemorrhagic** | `D:\final\mri\Hemorrhagic\` *(any slice)* |
| **MRI** | **Normal** | `D:\final\mri\Normal\` *(any slice)* |

---

## 📡 API Endpoints

- `GET /` — Serves the interactive clinical diagnostic dashboard.
- `GET /api/models` — Returns active model names, architectures, checkpoints, and weights.
- `GET /api/stats` — Returns total analyzed scans counter.
- `POST /api/analyze` — Multipart form upload (`file`, `modality`, `patient_name`, `patient_id`) returning predictions, probabilities, recommendations, and Grad-CAM URLs.

---

## ⚖️ Disclaimer

*NeuroScan is developed for academic, scientific, and research purposes. It serves as a clinical decision support tool and is not intended to replace professional evaluation by a licensed neurologist or radiologist.*
