# 🧪 Development & CLI Training Guide

## 1. Environment Self-Test & Unit Tests

Run the built-in hardware/dependency smoke test and the automated unit test suite from the project root:

```powershell
python main.py --selftest
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### What Is Tested (`tests/test_project_setup.py`)
- **`SettingsTemplateTests`**: Verifies `settings.example.json` seeding and `ensure_default_controls()` 16-button backfilling without overwriting custom user coordinates.
- **`PreviewBufferTests`**: Verifies lock-free latest-frame buffering and single-render guarantees in `BotGUI`.
- **`FarmingEnhancementsTests`**:
  - Full-base 360° storage deployment (including bottom quadrant storages) with 2D Surrender button bounding-box protection.
  - HSV saturation detection for skipping depleted (grayscale) troop cards on Wave 2+.
  - `Smart`, `AND`, and `OR` loot evaluation + 16-dimensional base feature extraction (`extract_base_features`).
  - `VideoAttackExtractor` `.mp4` frame analysis + `AttackPolicyNet` behavioral cloning & counterfactual Online RL (`learn_from_raid_outcome`).
  - `CoCCommanderNet` (`CoCTacticalBrain`) strategic planning, `BaseMemoryBank` topological k-NN recall (`AVOID_MISTAKE` & `REINFORCE_WIN`), and macro Online RL (`learn_from_raid`).

---

## 2. Training Neural Networks from the CLI

You can extract demonstrations from gameplay videos (`.mp4`) and retrain both `models/policy_net.pt` (`AttackPolicyNet`) and `models/coc_commander.pt` (`CoCCommanderNet`) either via the **`⚡ Retrain AI`** / **`🎬 Learn Attack (.mp4)`** buttons in the GUI or from the command line:

```powershell
# Extract human taps from a gameplay video and train PolicyNet + CommanderNet
.\.venv\Scripts\python.exe -m ai.video_learner --video "path\to\gameplay.mp4" --epochs 80

# Retrain both neural networks from existing dataset/attack_demos.jsonl & logs/raid_*.json
.\.venv\Scripts\python.exe -m ai.video_learner --train --epochs 100
```

---

## 3. Data & Model Artifacts

| File Path | Format | Description |
| :--- | :--- | :--- |
| `models/best.pt` | PyTorch / YOLO | Pre-trained Ultralytics YOLO weights for Clash of Clans buildings & UI |
| `models/coc_commander.pt` | PyTorch StateDict | Strategic 4-head neural network weights (`CoCCommanderNet`) |
| `models/policy_net.pt` | PyTorch StateDict | Micro perimeter tap predictor weights & learned strategy profile (`AttackPolicyNet`) |
| `dataset/base_memory_bank.json` | JSON | Associative k-NN topological memory bank (up to 512 base layouts) |
| `dataset/self_learned_raids.jsonl` | JSONL | Chronological log of Online RL reward, drain %, and mistake corrections |
| `dataset/attack_demos.jsonl` | JSONL | Extracted human gameplay trajectories from `.mp4` videos |
