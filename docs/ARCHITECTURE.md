# Architecture

## Overview

Coc-DeepFarm observes the emulator, decides when and how to attack using its **Dual-Brain PyTorch Deep Learning System (`CoCCommanderNet` + `AttackPolicyNet`)**, and injects taps without requiring the emulator window to be focused or visible.

Input goes through **ADB** (`adb shell input tap`). Live capture uses an H.264 stream from Android `screenrecord`, decoded by FFmpeg. Fixed menu buttons use **calibrated normalized coordinates** stored in local `settings.json` plus optional OpenCV template verification (`templates/*.png`). YOLO + High-Contrast EasyOCR locate changing base layouts and read resource values.

## Runtime data flow

```text
                                  ┌─ ADB exec-out screenrecord ─ FFmpeg
                                  │                                 │
main.py ── device.WindowManager ──┴── VideoCaptureStream ───────────┤
    │                                 (latest decoded frame)        │
    ├── core.CocBot worker ── capture_window() ─────────────────────┤
    │       ├── perception.VisionManager (YOLO / EasyOCR)           │
    │       ├── ai.CoCTacticalBrain (CoCCommanderNet macro plan)    │
    │       ├── ai.PolicyPlanner (AttackPolicyNet micro taps + RL)  │
    │       └── device.AutomationController ── ADB batch input      │
    │                                                               v
    └── ui.BotGUI ── preview reader ── latest-frame slot ── Tk render loop
                  └── event queue <── CocBot worker
```

## Clean Layered Package Structure

| Package / Module | Responsibility | Depends on |
|---|---|---|
| `main.py` | Application bootstrapper, `.venv` auto-switcher, `--selftest` | all packages |
| `core/config.py` | `AppConfig` dataclass (model path, ADB defaults, class names) | — |
| `core/settings.py` | Load/save `settings.json` seeded from `settings.example.json` | `core/utils.py` |
| `core/bot.py` | `CocBot` farming state machine, multi-wave deployment, online RL loop | `ai/`, `perception/`, `device/`, `services/` |
| `ai/brain.py` | `CoCCommanderNet` & `CoCTacticalBrain` (16-feature strategic neural net + online RL), `RuleBrain` | `torch`, `core/utils.py` |
| `ai/policy_net.py` | `AttackPolicyNet` & `PolicyPlanner` (9-feature perimeter tap generator + wave/raid RL) | `torch`, `core/utils.py` |
| `ai/video_learner.py` | `VideoAttackExtractor` (extracts human attack trajectories from `.mp4` videos) | `cv2`, `ai/brain.py`, `ai/policy_net.py` |
| `perception/vision.py` | YOLO object detection + dual-pass high-contrast EasyOCR | `torch`, `ultralytics`, `easyocr` |
| `perception/template_matcher.py` | Multi-scale OpenCV template matching for button verification | `cv2`, `numpy` |
| `device/adb.py` | ADB transport, subshell batch tapping, H.264 stream decoder | `subprocess`, `cv2`, FFmpeg |
| `device/window_manager.py` | Guest resolution, live frames, screenshot fallback | `device/adb.py` |
| `device/automation.py` | High-level `click`, `tap_many`, `hold` controller | `device/adb.py` |
| `services/logger.py` | Console & rotating file logging (`logs/bot.log`) | `logging` |
| `services/notify.py` | Push notifications via `ntfy.sh` with compressed screenshots | `urllib` |
| `services/replay.py` | Per-raid JSON telemetry (`logs/raid_*.json`) | `json` |
| `ui/gui.py` | Dark-mode Tkinter dashboard, live telemetry, video trainer trigger | `tkinter`, `PIL`, `ai/` |
| `ui/calibrate.py` | Interactive click/drag button calibrator & template cropper | `tkinter`, `PIL`, `device/adb.py` |

## Dual-Brain Deep Learning & Online Reinforcement Learning

1. **Pre-Raid Macro Decision (`CoCTacticalBrain.plan`)**:
   - Extracts a 16-dimensional feature vector (`extract_base_features`) from OCR loot, target ratios, storage quadrant clustering, outer-ring collector exposure, and spread.
   - `CoCCommanderNet` (`models/coc_commander.pt`) predicts `expected_profit`, `target_side` (`all`, `left`, `right`), `style_idx` (combo order vs fast goblin blitz), and `surrender_after_s` in `< 1 ms` on GPU.
2. **In-Battle Micro Execution (`PolicyPlanner.predict_points`)**:
   - `AttackPolicyNet` (`models/policy_net.pt`) predicts deployment offsets `(dx, dy)` and inter-tap delays `dt` for each storage, projecting coordinates onto the outer perimeter ring (`min_deploy_radius`) and avoiding the Surrender button bounding box.
   - Between waves, `CocBot._filter_active_slots` checks HSV color saturation of each troop card to skip depleted (grayscale) slots.
   - `PolicyPlanner.record_wave_feedback` compares detected storages before and after each wave (`+1.0` reward for destroyed storages, `-0.5` penalty for surviving storages).
3. **Post-Raid Online Self-Learning (`learn_from_raid_outcome` & `learn_from_raid`)**:
   - Computes realized reward from post-battle loot gain and storage drain ratio.
   - Reinforces successful tap offsets and strategic choices, or applies counterfactual outward-perimeter & flank-tangent corrections when a raid underperforms, saving updated weights immediately to `models/policy_net.pt` and `models/coc_commander.pt`.
