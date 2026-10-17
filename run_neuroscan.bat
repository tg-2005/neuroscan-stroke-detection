@echo off
title NeuroScan - Deep Learning Stroke Detection System
color 0B

echo ======================================================================
echo           NEUROSCAN: EARLY STROKE DETECTION SYSTEM
echo       Dual-Stream CNN-Transformer Architecture with Grad-CAM
echo ======================================================================
echo.

set SCRIPT_DIR=%~dp0
set NEUROSCAN_DIR=%SCRIPT_DIR%sproject\neuroscan

:: Find Python: test standard python command first, then fallback to explicit path
set PYTHON_EXE=python
%PYTHON_EXE% --version >nul 2>&1
if %errorlevel% neq 0 (
    set PYTHON_EXE=C:\Users\TARANG\AppData\Local\Programs\Python\Python311\python.exe
)

echo [*] Checking Python environment...
"%PYTHON_EXE%" --version
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not found!
    pause
    exit /b 1
)

echo [*] Python verified successfully.
echo [*] Checking AI Model Checkpoints...

if exist "%SCRIPT_DIR%mri\outputs\run_hybrid_cbam_swin\best_model.pt" (
    echo     [+] MRI Model : Hybrid CNN-Transformer - READY
) else (
    echo     [-] MRI Model : Using default fallback
)

if exist "%SCRIPT_DIR%preprocessed\outputs\hybrid_cbam_swin\best_model.pt" (
    echo     [+] CT Model  : 3-Model Ensemble - READY
) else (
    echo     [-] CT Model  : 2-Model Ensemble fallback
)

echo.
echo [*] Project Workspace: %SCRIPT_DIR%
echo [*] Web Application : %NEUROSCAN_DIR%
echo.
echo [*] Launching NeuroScan Flask Backend on http://localhost:5000...
echo [*] Press Ctrl+C in this terminal window to stop the server.
echo.

cd /d "%NEUROSCAN_DIR%"
start "" http://localhost:5000
"%PYTHON_EXE%" app.py

pause
