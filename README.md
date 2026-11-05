# Coc-DeepFarm

<p align="center">
  <strong>AI & Deep Learning Clash of Clans Farming Automation for Android Emulators</strong><br>
  YOLO Vision · High-Contrast OCR · Behavioral Cloning (`AttackPolicyNet`) · Local VLM/LLM (Ollama) & Gemini AI · ADB Stream
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#deep-learning--ai-brain">Deep Learning & AI Brain</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#project-structure">Project Structure</a>
</p>

---

## Overview

**Coc-DeepFarm** is a Clean-Layered Python desktop application that autonomously plays and farms in Clash of Clans through an Android emulator. It combines computer vision (**YOLO** + **Template Matching** + **EasyOCR**), a **Two-Tier AI Architecture** (Strategic LLM/VLM Brain + Deep Learning Behavioral Cloning Policy Network), and low-latency **ADB** control.

All screen capture and touch inputs run directly over ADB in the background, meaning the bot does not hijack your desktop mouse cursor or require the emulator window to remain focused.

## Features

- **Clean Layered Package Architecture** — Strictly separated domain packages (`core/`, `ai/`, `perception/`, `device/`, `services/`, `ui/`).
- **Learn from Gameplay Video (`.mp4`)** — Feed recorded attack videos into `VideoAttackExtractor` to train `AttackPolicyNet` (PyTorch CUDA) via Behavioral Cloning so troop placement mimics human tap offsets and timing.
- **3-Mode Strategic Brain** — Switch seamlessly in the GUI between:
  1. **Rule-based (`RuleBrain`)** — Fast deterministic geometry and dead-base heuristics.
  2. **Local AI (`LocalLLMBrain`)** — 100% local multimodal VLM/LLM via **Ollama** (e.g., `qwen2.5vl:3b` on RTX GPUs) with zero API quota limits.
  3. **Gemini Cloud AI (`GeminiBrain`)** — Google Gemini (`gemini-2.5-flash` with automatic `gemini-2.5-flash-lite` fallback and quota-saving pre-filters).
- **Smart Loot Filter & Dead-Base Detection** — Cycle between `Smart` (Combo Gold+Elixir, Single-Resource Jackpot, and outer-ring Dead-Base collector boost), `Strict (AND)`, and `Flexible (OR)` directly from the dashboard.
- **Full-Base 360° Deployment & Depleted Slot Skipping** — Deploys across the entire map (including bottom storages) using a 2D bounding-box guard around the Surrender button, and measures HSV color saturation to automatically skip depleted (grayscale) troop slots in Wave 2+.
- **High-Contrast Post-Battle OCR** — Dual-pass OCR with binary thresholding (`195`) + morphological closing and consecutive-poll stabilization to prevent `0` loot misreads during post-battle counter animations.
- **Parallel ADB Batch Tapping & H.264 Stream** — Subshell-batched ADB multi-taps (~3.5× faster troop waves) and 20 FPS H.264 live emulator stream decoding.
- **Live Monitoring Dashboard** — Real-time annotated preview, YOLO detection table, state stepper, session KPIs, latency telemetry graph, and **F12** emergency stop hotkey.

---

## Architecture

<p align="center">
  <img src="docs/Arsitektur%20Sistem%20AI%20Coc-DeepFarm.png" alt="Coc-DeepFarm system architecture" width="100%">
</p>

The codebase is organized into six domain layers following **Clean Layered Architecture**:

| Layer / Package | Modules | Responsibility |
|---|---|---|
| **Entry Point** | `main.py` | Auto-activates `.venv`, initializes ADB & H.264 stream, launches Tkinter GUI |
| **`core/`** | `bot.py`, `config.py`, `settings.py`, `utils.py` | Main farming state machine, multi-wave deployment, recovery, runtime settings |
| **`ai/`** | `brain.py`, `policy_net.py`, `video_learner.py` | `RuleBrain`, `LocalLLMBrain` (Ollama), `GeminiBrain`, PyTorch `AttackPolicyNet`, `.mp4` video demo extractor |
| **`perception/`** | `vision.py`, `template_matcher.py` | YOLO object detection, dual-pass EasyOCR, OpenCV template verification |
| **`device/`** | `adb.py`, `window_manager.py`, `automation.py` | ADB transport, parallel batch taps, H.264 video stream reader |
| **`services/`** | `logger.py`, `notify.py`, `replay.py`, `replay_report.py` | Structured logging, `ntfy` push notifications with screenshots, raid JSON telemetry |
| **`ui/`** | `gui.py`, `calibrate.py` | Live dark-mode dashboard, strategy/filter toggles, video trainer trigger, interactive button calibration |

---

## Deep Learning & AI Brain

Coc-DeepFarm separates **Macro Strategy** from **Micro Motor Execution** so you can mix and match any strategic brain with video-learned troop deployment:

### 1. Video Behavioral Cloning (`AttackPolicyNet`)
- Click **`🎬 Learn Attack (.mp4)`** in the GUI (or run `python -m ai.video_learner --video <path.mp4>`).
- `VideoAttackExtractor` scans the video to detect resource storage positions, bottom troop-bar slot switches, and battlefield troop spawn bursts `(x, y, dt)`.
- `train_policy_from_dataset()` trains a PyTorch neural network (`AttackPolicyNet`) on your GPU (`models/policy_net.pt`) to predict human-like deployment offsets relative to enemy storages.
- Once trained, `CocBot._deploy_targets` automatically uses `PolicyPlanner` across all strategy modes (`Rule-based + PolicyNet`, `Local AI + PolicyNet`, or `Gemini Cloud AI + PolicyNet`).

### 2. Strategic Brain Modes (`ai/brain.py`)
- **Gemini Cloud AI (`name: "gemini"`)**: Uses your `GEMINI_API_KEY` with `gemini-2.5-flash` and automatic fallback to `gemini-2.5-flash-lite`. Includes `should_consult_ai()` pre-filtering so obviously empty bases are skipped immediately without burning API quota.
- **Local VLM / LLM (`name: "local_llm"`)**: Connects to a local [Ollama](https://ollama.com) instance (`http://127.0.0.1:11434/api/generate`, default model `qwen2.5vl:3b`). Sends both the live scouting screenshot (Base64) and base geometry JSON to decide `skip_base`, `target_side` (`left`/`right`/`all`), `deploy_order`, and `surrender_after_s`.
- **Rule-Based (`name: "rule"`)**: Deterministic evaluation with dead-base collector ring detection and dominant storage-cluster side inference.

---

## Getting Started

### Requirements
- Windows 10/11 with Python 3.10+ (Python 3.12 recommended)
- Android emulator with ADB enabled (MuMu Player 12, LDPlayer, BlueStacks, or MEmu)
- `ffmpeg` on `PATH` (optional, enables 20 FPS H.264 live stream; falls back to ADB screenshots)
- NVIDIA GPU (e.g., RTX 30/40 series) recommended for CUDA-accelerated YOLO, EasyOCR, `AttackPolicyNet`, and optional Ollama Local VLM

### 1. Create Virtual Environment & Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA GPU acceleration (PyTorch CUDA 12.8):

```powershell
pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Calibrate Emulator Controls (First Run)

Open Clash of Clans on the village screen in your emulator and run:

```powershell
python ui/calibrate.py
```

Click or drag a bounding box around each prompted button. Coordinates and reference templates (`templates/*.png`) are saved automatically.

### 3. Run the Bot

```powershell
python main.py
```
*(Note: `main.py` and `ui/calibrate.py` automatically switch to `.venv\Scripts\python.exe` even if invoked from global Python, or you can double-click `run.bat`.)*

### 4. Train from Attack Gameplay Video (Optional)

Either click **`🎬 Learn Attack (.mp4)`** inside the GUI, or run from the terminal:

```powershell
python -m ai.video_learner --video "path\to\your_attack.mp4" --epochs 80
```

### 5. Run Self-Test & Unit Tests

```powershell
python main.py --selftest
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## Configuration

Runtime configuration is stored in `settings.json` (auto-created from `settings.example.json` on first launch):

| Section | Key Options | Description |
|---|---|---|
| `controls` | `attack_btn_home`, `find_btn`, `nxt_btn`, `s_goblin`, ... | Normalized `(x, y)` button coordinates from `ui/calibrate.py` |
| `attack` | `target_mode` (`"smart"` \| `"and"` \| `"or"`), `drain_ratio`, `time_limit` | Loot filter mode, storage drain target (default `0.8`), max battle duration |
| `troops` | `slots`, `deploy_mode` (`"target"` \| `"tap"` \| `"hold"`), `skip_empty_slots` | Troop bar slots, deployment algorithm, HSV grayscale empty-slot skipping |
| `brain` | `name` (`"gemini"` \| `"local_llm"` \| `"rule"`), `model`, `local_model`, `use_ai` | AI planner selection, Gemini/Ollama model names, fallback model |
| `waits` | `click_verify`, `template_match_threshold`, `scan_base_first` | Smart visual click verification and scan timing |
| `notifications` | `enabled`, `server`, `topic`, `token` | Optional `ntfy` push notifications |

---

## Project Structure

```text
Coc-Auto-Farm/
├── main.py                      # Application entry point & --selftest
├── run.bat                      # One-click Windows launcher & environment bootstrapper
├── settings.example.json        # Starter configuration template
├── core/                        # Core Domain & Orchestration Layer
│   ├── bot.py                   # CocBot farming state machine & multi-wave deployment
│   ├── config.py                # AppConfig static defaults & YOLO class constants
│   ├── settings.py              # Settings loader & atomic saver
│   └── utils.py                 # Path resolution & CUDA runtime setup
├── ai/                          # Artificial Intelligence & Deep Learning Layer
│   ├── brain.py                 # RuleBrain, GeminiBrain (Cloud API), LocalLLMBrain (Ollama VLM)
│   ├── policy_net.py            # PyTorch AttackPolicyNet (Behavioral Cloning on GPU)
│   └── video_learner.py         # VideoAttackExtractor (.mp4 gameplay -> training dataset)
├── perception/                  # Computer Vision & OCR Layer
│   ├── vision.py                # Ultralytics YOLO detector + High-Contrast EasyOCR
│   └── template_matcher.py      # Multi-scale OpenCV template matching & ROI verifier
├── device/                      # Emulator Transport & Input Layer
│   ├── adb.py                   # ADB controller, parallel batch tap, H.264 stream decoder
│   ├── window_manager.py        # Frame capture manager & latency tracker
│   └── automation.py            # High-level click, tap_many, and long-press controller
├── services/                    # Telemetry, Logging & Notification Layer
│   ├── logger.py                # Rotating file & console logger
│   ├── notify.py                # Asynchronous ntfy push notifications
│   ├── replay.py                # Per-raid JSON state/event/outcome recorder
│   └── replay_report.py         # CLI raid analytics & AI agreement reporter
├── ui/                          # Presentation Layer
│   ├── gui.py                   # Live Tkinter dashboard, telemetry graphs & AI controls
│   └── calibrate.py             # Interactive button coordinate & template calibrator
├── models/                      # YOLO weights (best.pt) & learned policy (policy_net.pt)
├── templates/                   # Cropped reference button images for visual verification
└── tests/                       # Unit & regression test suite
    └── test_project_setup.py
```

---

## License

MIT — see [LICENSE](LICENSE).
