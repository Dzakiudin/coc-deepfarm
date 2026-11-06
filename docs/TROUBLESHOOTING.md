# 🛠️ Troubleshooting Guide

## 1. Emulator & ADB Connection

### "Emulator Belum Terdeteksi" / ADB Not Connected
> [!TIP]
> You no longer need to restart the bot if you forgot to open your emulator first!

1. Start your Android emulator (**MuMu Player 12**, **LDPlayer**, **BlueStacks**, or **MEmu**).
2. Click **`🔌 Connect ADB`** in the top toolbar of the dashboard (or simply click **`▶ Start Farming`**—the Pre-Flight Check will connect automatically).
3. If using **LDPlayer** or **BlueStacks**, ensure **ADB Debugging / Local Connection** is enabled in the emulator's settings.

---

## 2. Button Clicks & Resolution

### Buttons Click Slightly Off-Target on a Custom Aspect Ratio
- Standard 16:9 resolutions (`1280x720`, `1600x900`, `1920x1080`, `1680x720`) work automatically using the built-in `DEFAULT_CONTROLS` preset.
- If your emulator uses a non-standard aspect ratio:
  1. Click **`🎯 Calibrate`** in the dashboard header.
  2. Click or drag a bounding box around the specific button you want to adjust (or click **Reset Preset** to restore defaults).
  3. Click **Save to settings**.

---

## 3. Vision & OCR Scanning

### "Could not find [class] after X attempts"
- Ensure Clash of Clans is open on your main village screen before pressing **`▶ Start Farming`**.
- If your internet or emulator loads clouds slowly, increase `waits.scan_base_first` (default `4.0` seconds) or `recovery.scan_restart_limit` in [`settings.json`](../settings.json).

---

## 4. GPU / CUDA Acceleration Check

Verify that PyTorch CUDA, YOLO, and EasyOCR are running on your NVIDIA GPU at any time with:

```powershell
python main.py --selftest
```
