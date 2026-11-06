# 📦 Changelog & Release History

All notable changes to **Coc-DeepFarm** are documented in this file.

---

## [v2.0.0] — Dual-Brain DeepRL, Associative Base Memory & 1-Click Zero-Config Edition (2026-11-06)

### 🧠 100% Dedicated CoC Dual-Brain Deep Learning (`ai/`)
- **Replaced External Cloud/Local Text LLMs (`Gemini` / `Ollama`)** with a 100% native PyTorch CUDA Dual-Brain architecture running in **`< 1 ms`** per decision with zero API keys or rate limits:
  - **`CoCCommanderNet` (`ai/brain.py`)**: 16-feature multi-head strategic neural network (`models/coc_commander.pt`) predicting expected stealable profit ratio (`profit_head`), optimal attack flank (`side_head`: `all`/`left`/`right`), troop deployment doctrine (`style_head`), and dynamic surrender timer (`timer_head`: `45s–140s`).
  - **`AttackPolicyNet` (`ai/policy_net.py`)**: 9-feature micro-tactical neural network (`models/policy_net.pt`) predicting human-like perimeter offsets `(dx, dy)`, inter-tap delays `dt`, and per-slot tap budgets relative to detected enemy storages.
- **Associative Topological Base Memory (`BaseMemoryBank` in `ai/brain.py`)**:
  - Stores up to **512 encountered enemy base layouts** (`dataset/base_memory_bank.json`) and recalls twin bases ($\ge 92\%$ weighted Cosine Similarity) via **`REINFORCE_WIN`** (locks in proven winning strategies) and **`AVOID_MISTAKE`** (blocks flanks/styles that previously failed on that topology).
  - Consolidates near-identical layouts ($\ge 97.5\%$ similarity) via Exponential Moving Average (`hits + 1`).
- **Autonomous Online Reinforcement Learning (Self-Learning from Mistakes)**:
  - Per-wave storage destruction feedback (`+1.0` reward for destroyed storages, `-0.5` penalty for survivors).
  - Post-raid counterfactual gradient updates (`learn_from_raid_outcome` & `learn_from_raid`) backed by a **256-sample Experience Replay Buffer**, gradient norm clipping (`max_norm=0.5`), and adaptive learning-rate decay.
- **Gameplay Video Demonstration Learner (`ai/video_learner.py`)**:
  - Extracts human tap bursts, troop slot sequences, and storage layouts directly from `.mp4` gameplay videos via **`🎬 Learn Attack (.mp4)`**.

### 🖥️ Enhanced v2.0 DeepRL Dashboard (`ui/gui.py`)
- **👁️ Live AI Tactical Radar Overlay**: Draws the learned safe deployment perimeter ellipse (`min_deploy_radius`), planned tap crosshairs `(⊕)`, and a real-time HUD (`PROFIT %`, `FLANK`, `MEM % WIN/FIX/NEW`, `DEAD BASE`) directly on the live emulator stream.
- **📈 Dual-Mode Telemetry Chart (`🧠 AI Curve ⇄ Latency`)**: Switches between the live **Online RL Self-Learning Curve** (`Reward %`, `Drain %`, `Reinforce` vs `Mistake Fix`) and the **System Latency Graph (`Capture / YOLO / OCR ms`)**.
- **⚡ Interactive Quick-Action Toolbar**: Added 1-click **`🔌 Connect ADB`**, **`⚡ Retrain AI`**, **`🎯 Calibrate`**, and **`📸 Snapshot`** buttons plus live neural episode & memory counters.
- **⏱️ Hourly Farming Rate KPIs**: Displays real-time hourly loot rates (`M / hr`) and average storage drain percentage (`Avg Drain %`), plus 1-click minimum loot presets (`400k`, `700k`, `1M+`).

### 🚀 Beginner-Friendly 1-Click Zero-Config Onboarding
- **`Start-Bot.bat` (1-Click Smart Launcher)**: Automatically creates `.venv`, detects NVIDIA GPUs (`nvidia-smi`), installs CUDA or CPU dependencies, and launches the dashboard with a single double-click.
- **Universal 16:9 CoC 16-Button Control Preset (`DEFAULT_CONTROLS` in `core/settings.py`)**: Works out-of-the-box on standard landscape resolutions (`1280x720`, `1600x900`, `1920x1080`, `1680x720`) without requiring manual 16-button calibration, plus a **Reset Preset** button in `ui/calibrate.py`.
- **Non-Blocking ADB & Pre-Flight Auto-Check (`main.py`, `ui/gui.py`)**: Opening the bot before starting the emulator no longer exits; clicking **`▶ Start Farming`** automatically connects ADB, starts the live video stream, and backfills any missing button coordinates.

---

## [v1.0.0] — Initial Public Release (2026-10-29)

- ADB-based emulator input and live H.264 frame capture with ADB screenshot fallback.
- YOLO object detection (`models/best.pt`) and EasyOCR loot reading with optional CUDA acceleration.
- Initial rule-based strategy and Tkinter monitoring dashboard.
