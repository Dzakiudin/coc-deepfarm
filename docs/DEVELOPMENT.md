# Development

## Validation & Testing

Run the project's dependency/model smoke test and unit test suite from the repository root:

```powershell
python main.py --selftest
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- `--selftest` verifies OpenCV, NumPy, Pillow, Keyboard, PyTorch CUDA (`RTX 4060`), TorchVision, Ultralytics YOLO (`models/best.pt`), and EasyOCR.
- `tests/test_project_setup.py` tests settings seeding, preview frame buffering, bottom-base storage deployment & surrender button protection, HSV grayscale empty-slot skipping, smart loot evaluation, `VideoAttackExtractor`, `AttackPolicyNet` (`PolicyPlanner`), `CoCCommanderNet` (`CoCTacticalBrain`), and online reinforcement learning from both raid mistakes and successes.

## Training the CoC Deep Learning Models from CLI

You can extract demonstrations from gameplay videos (`.mp4`) and retrain both `models/policy_net.pt` (`AttackPolicyNet`) and `models/coc_commander.pt` (`CoCCommanderNet`) directly from the command line:

```powershell
.\.venv\Scripts\python.exe -m ai.video_learner --video "path\to\gameplay.mp4" --epochs 80
```

Or retrain both neural networks from existing demonstrations in `dataset/attack_demos.jsonl` and raid logs in `logs/raid_*.json`:

```powershell
.\.venv\Scripts\python.exe -m ai.video_learner --train --epochs 100
```

## Module Overview

- **`core/bot.py` (`CocBot`)**: Orchestrates the farming loop, multi-wave troop deployment, smart-click verification, and triggers `policy_planner.record_wave_feedback()`, `policy_planner.learn_from_raid_outcome()`, and `brain_alt.learn_from_raid()`.
- **`ai/brain.py` (`CoCCommanderNet`, `CoCTacticalBrain`, `RuleBrain`)**: 16-feature strategic neural network and online reinforcement learner.
- **`ai/policy_net.py` (`AttackPolicyNet`, `PolicyPlanner`)**: 9-feature perimeter tap coordinate predictor and wave/raid counterfactual RL updater.
- **`ai/video_learner.py` (`VideoAttackExtractor`)**: Extracts storage positions, active troop slot selections, and spawn tap bursts from `.mp4` videos.
- **`perception/vision.py` (`VisionManager`)**: YOLO object detection + dual-pass high-contrast EasyOCR.
- **`device/adb.py` (`EmulatorAdb`, `VideoCaptureStream`)**: ADB connection, subshell batch tapping, and FFmpeg H.264 stream decoding.
- **`ui/gui.py` (`BotGUI`)** & **`ui/calibrate.py` (`Calibrator`)**: Tkinter dashboard and interactive calibration tool.
