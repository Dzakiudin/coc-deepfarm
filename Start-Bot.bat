@echo off
setlocal EnableDelayedExpansion
title Coc-DeepFarm - 1-Click Smart Launcher
cd /d "%~dp0"

echo ============================================================
echo   Coc-DeepFarm - Dedicated CoC Deep Learning Auto-Farm
echo ============================================================
echo.

if exist ".venv\Scripts\python.exe" (
    goto :LAUNCH
)

echo [Setup Pertama Kali] Virtual environment (.venv) belum ditemukan.
echo Menyiapkan instalasi otomatis agar siap pakai...
echo.

where python >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set "SYS_PY=python"
) else (
    where py >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set "SYS_PY=py -3"
    ) else (
        echo [ERROR] Python 3.10+ belum terinstal di Windows kamu.
        echo Silakan unduh Python dari https://www.python.org/downloads/
        echo ^(Jangan lupa centang "Add Python to PATH" saat instalasi^).
        echo.
        pause
        exit /b 1
    )
)

echo [1/3] Membuat lingkungan virtual (.venv)...
%SYS_PY% -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Gagal membuat .venv.
    pause
    exit /b 1
)

echo [2/3] Memperbarui pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [3/3] Mendeteksi GPU NVIDIA (CUDA)...
nvidia-smi >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       GPU NVIDIA terdeteksi! Menginstal paket akselerasi CUDA + AI...
    ".venv\Scripts\pip.exe" install -r requirements-gpu.txt
    ".venv\Scripts\pip.exe" install -r requirements.txt
) else (
    echo       Menginstal paket standar (CPU/Universal)...
    ".venv\Scripts\pip.exe" install -r requirements.txt
)

echo.
echo [SUKSES] Instalasi selesai! Membuka Coc-DeepFarm...
echo.

:LAUNCH
echo Menjalankan Coc-DeepFarm...
".venv\Scripts\python.exe" main.py %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [INFO] Aplikasi berhenti dengan kode keluar %ERRORLEVEL%.
    pause
)
endlocal
