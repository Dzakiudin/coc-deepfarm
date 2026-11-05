# Coc-DeepFarm

<p align="center">
  <strong>Dedicated Deep Learning & Online Reinforcement Learning Clash of Clans Farming Bot</strong><br>
  YOLO Vision · High-Contrast OCR · `CoCCommanderNet` (Strategic Brain) · `AttackPolicyNet` (Behavioral Cloning + Online RL) · ADB H.264 Stream
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#dual-brain-deep-learning--online-rl">Dual-Brain Deep Learning & Online RL</a> ·
  <a href="#getting-started">Getting Started</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#project-structure">Project Structure</a>
</p>

---

## Overview

**Coc-DeepFarm** is a Clean-Layered Python desktop application that autonomously plays, learns, and farms in Clash of Clans through an Android emulator. Unlike generic bots that rely on rigid rules or slow/rate-limited external text LLMs, Coc-DeepFarm runs a **100% Dedicated Clash of Clans Dual-Brain Neural Network (`CoCCommanderNet` + `AttackPolicyNet`)** natively on your GPU (PyTorch CUDA) in under `1 ms` per decision—and **automatically learns from its own mistakes and successes after every wave and every raid**.

All screen capture and touch inputs run directly over ADB in the background, meaning the bot does not hijack your desktop mouse cursor or require the emulator window to remain focused.

## Features

- **Clean Layered Package Architecture** — Strictly separated domain packages (`core/`, `ai/`, `perception/`, `device/`, `services/`, `ui/`).
- **100% Native CoC Dual-Brain Deep Learning (Zero API Key, Zero Ollama Overhead)**:
  1. **`CoCCommanderNet` (`ai/brain.py`)** — Multi-head strategic neural network (`models/coc_commander.pt`) that encodes 16 spatial & resource features of an enemy base to predict stealable profit ratio, optimal attack direction (`all`, `left`, `right`), troop deployment doctrine, and dynamic surrender timer in `< 1 ms` on GPU.
  2. **`AttackPolicyNet` (`ai/policy_net.py`)** — Micro-tactical neural network (`models/policy_net.pt`) that predicts human-like perimeter tap offsets `(dx, dy)`, timing delays `dt`, and per-slot tap budgets relative to detected enemy storages.
- **Learn from Gameplay Video (`.mp4`)** — Feed recorded attack videos into `VideoAttackExtractor` via the **`🎬 Learn Attack (.mp4)`** button to train both `CoCCommanderNet` and `AttackPolicyNet` on your personal playstyle.
- **Autonomous Online Reinforcement Learning (Self-Learning from Mistakes)** — Evaluates storage destruction after every wave (`+1.0` reward for destroyed storages, `-0.5` penalty for surviving storages) and executes counterfactual gradient updates after every raid (`dataset/self_learned_raids.jsonl`).
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
| **`core/`** | `bot.py`, `config.py`, `settings.py`, `utils.py` | Main farming state machine, multi-wave deployment, online RL triggers, runtime settings |
| **`ai/`** | `brain.py`, `policy_net.py`, `video_learner.py` | PyTorch `CoCCommanderNet` (`CoCTacticalBrain`), `AttackPolicyNet` (`PolicyPlanner`), `.mp4` video demo extractor |
| **`perception/`** | `vision.py`, `template_matcher.py` | YOLO object detection, dual-pass EasyOCR, OpenCV template verification |
| **`device/`** | `adb.py`, `window_manager.py`, `automation.py` | ADB transport, parallel batch taps, H.264 video stream reader |
| **`services/`** | `logger.py`, `notify.py`, `replay.py` | Structured logging, `ntfy` push notifications with screenshots, raid JSON telemetry |
| **`ui/`** | `gui.py`, `calibrate.py` | Live dark-mode dashboard, strategy/filter toggles, video trainer trigger, interactive button calibration |

---

## Dual-Brain Deep Learning & Online RL

Coc-DeepFarm uses two specialized PyTorch neural networks that work together and continuously self-improve:

### 1. Macro Strategist: `CoCCommanderNet` (`ai/brain.py`)
- Encodes the enemy base into a 16-dimensional spatial & resource vector (`extract_base_features`): resource-to-target ratios, dead-base collector indicators, average storage distance from center, quadrant densities (`left`, `right`, `top`, `bottom`), outer-ring exposure fraction, and spatial spread.
- Outputs 4 tactical heads:
  - **`profit_head`**: Expected fraction of loot stealable from the base layout.
  - **`side_head`**: Optimal attack vector (`all`, `left`, `right`).
  - **`style_head`**: Troop deployment doctrine (learned combo sequence vs fast goblin collector blitz).
  - **`timer_head`**: Dynamic surrender timer (`45s`–`140s`).

### 2. Micro Tactician: `AttackPolicyNet` (`ai/policy_net.py`)
- Takes a 9-dimensional per-storage feature vector (`[sx, sy, out_vx, out_vy, dist, is_gold, is_elixir, k_norm, t_norm]`) and predicts exact perimeter deployment offsets `(dx, dy)` and inter-tap delays `dt`.
- Projects all coordinates onto the outer grass perimeter ring (`min_deploy_radius`) so troops never land inside the red no-deploy zone.

### 3. Online Self-Learning from Mistakes
- **Per-Wave Feedback (`record_wave_feedback`)**: Compares detected storages before and after each wave. Destroyed storages receive `local_reward = +1.0`; surviving storages receive `local_reward = -0.5`.
- **Post-Raid Backpropagation (`learn_from_raid_outcome` & `learn_from_raid`)**:
  - When a raid succeeds (`REINFORCE SUKSES`), both neural networks reinforce the winning offsets, attack side, and troop sequence.
  - When a raid underperforms (`KOREKSI KESALAHAN`), `PolicyPlanner` computes counterfactual outward-perimeter & flank-tangent target corrections, expands `min_deploy_radius`, rotates underperforming lead troop slots, and updates `models/policy_net.pt` and `models/coc_commander.pt` on the spot.

---

## Getting Started

### Requirements
- Windows 10/11 with Python 3.10+ (Python 3.12 recommended)
- Android emulator with ADB enabled (MuMu Player 12, LDPlayer, BlueStacks, or MEmu)
- `ffmpeg` on `PATH` (optional, enables 20 FPS H.264 live stream; falls back to ADB screenshots)
- NVIDIA GPU (e.g., RTX 30/40 series) recommended for CUDA-accelerated YOLO, EasyOCR, `CoCCommanderNet`, and `AttackPolicyNet`

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
| `brain` | `name` (`"coc_ai"` \| `"rule"`), `use_ai` | Toggle between `CoCCommanderNet` Deep AI (`"coc_ai"`) and deterministic `"rule"` |
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
│   ├── bot.py                   # CocBot farming state machine, multi-wave deployment & RL loop
│   ├── config.py                # AppConfig static defaults & YOLO class constants
│   ├── settings.py              # Settings loader & atomic saver
│   └── utils.py                 # Path resolution & CUDA runtime setup
├── ai/                          # Dedicated CoC Deep Learning & Online RL Layer
│   ├── brain.py                 # PyTorch CoCCommanderNet (CoCTacticalBrain) & RuleBrain
│   ├── policy_net.py            # PyTorch AttackPolicyNet (Behavioral Cloning + Online RL)
│   └── video_learner.py         # VideoAttackExtractor (.mp4 gameplay -> neural training)
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
│   └── replay.py                # Per-raid JSON state/event/outcome recorder
├── ui/                          # Presentation Layer
│   ├── gui.py                   # Live Tkinter dashboard, telemetry graphs & AI controls
│   └── calibrate.py             # Interactive button coordinate & template calibrator
├── models/                      # YOLO (best.pt), Strategic Brain (coc_commander.pt) & PolicyNet (policy_net.pt)
├── dataset/                     # Extracted video demos (attack_demos.jsonl) & RL history (self_learned_raids.jsonl)
├── templates/                   # Cropped reference button images for visual verification
└── tests/                       # Unit & regression test suite
    └── test_project_setup.py
```

---

## License

MIT — see [LICENSE](LICENSE).
