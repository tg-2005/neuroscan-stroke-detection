#!/bin/bash
# NeuroScan — Startup Script
# Run this from the neuroscan/ directory

echo "========================================="
echo "  NeuroScan — Stroke Detection System"
echo "========================================="
echo ""

# Check Python
python3 --version || { echo "ERROR: Python 3 not found"; exit 1; }

# Install dependencies (skip if already installed)
echo "📦 Installing dependencies..."
pip install flask flask-cors numpy Pillow opencv-python-headless --break-system-packages -q

# Try to install PyTorch (CT model)
echo "🔦 Installing PyTorch (for CT/GoogLeNet model)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --break-system-packages -q 2>/dev/null || \
pip install torch torchvision --break-system-packages -q 2>/dev/null || \
echo "⚠️  PyTorch not installed — CT model will run in demo mode"

# Try to install TensorFlow (MRI model)
echo "🧠 Installing TensorFlow (for MRI/ResNet-50 model)..."
pip install tensorflow --break-system-packages -q 2>/dev/null || \
echo "⚠️  TensorFlow not installed — MRI model will run in demo mode"

echo ""
echo "✅ Checking model files..."
[ -f "models/stroke_model_final.pth" ] && echo "   ✅ CT model (stroke_model_final.pth) found" || echo "   ⚠️  CT model NOT found — will use demo mode"
[ -f "models/stroke_resnet50_v1.h5"  ] && echo "   ✅ MRI model (stroke_resnet50_v1.h5) found"  || echo "   ⚠️  MRI model NOT found — will use demo mode"

echo ""
echo "🚀 Starting Flask server on http://localhost:5000"
echo "   Open your browser and go to: http://localhost:5000"
echo ""
python3 app.py
