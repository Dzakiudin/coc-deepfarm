# Usage

## Setup Checklist

1. Android emulator running Clash of Clans on your main village with ADB enabled.
2. Python virtual environment `.venv` created with `requirements.txt` (and `requirements-gpu.txt` for NVIDIA CUDA).
3. YOLO weights present at `models/best.pt`.
4. FFmpeg on `PATH` for 20 FPS H.264 live video streaming (falls back to ADB screenshots if absent).
5. Buttons calibrated via `python ui/calibrate.py`.

## 1. Calibrate Buttons (`ui/calibrate.py`)

```powershell
python ui/calibrate.py
```

1. The tool connects via ADB and displays the current emulator screen.
2. Navigate the game to each screen (Main Village → Find Match → Scouting → In-Battle → Surrender Dialog → Post-Battle Return Home) and press **Recapture** to refresh the screenshot.
3. Click or drag a box around each prompted button (`attack_btn_home`, `find_btn`, `nxt_btn`, `attack_btn`, `s_goblin`, `troop_slot_1..5`, `surr`, `surr_conf`, `return_home`, `retry_btn_1..3`).
4. Click **Save** to write normalized coordinates to `settings.json` and save reference button templates to `templates/`.

## 2. Run the Bot (`main.py`)

```powershell
python main.py
```
*(Or double-click `run.bat`.)*

## 3. Dashboard Controls

| Control | Description |
|---|---|
| **Target per Raid (Gold / Elixir / Dark)** | Minimum resources required on an enemy base |
| **Target Total (Session)** | Session farming goals before the bot stops and notifies you |
| **Strategy Mode** | Toggle between **`CoC Deep AI (Commander + PolicyNet)`** (PyTorch GPU neural networks) and **`Rule-based + PolicyNet`** |
| **Loot Filter** | Cycle between **`Smart (Combo + Dead Base)`**, **`Strict (AND)`**, and **`Flexible (OR)`** |
| **🎬 Learn Attack (.mp4)** | Extract human tap trajectories & storage layouts from a gameplay video and train both `CoCCommanderNet` and `AttackPolicyNet` |
| **▶ Start Farming / ■ Stop (F12)** | Start or stop the autonomous farming loop (`F12` is the global emergency stop hotkey) |

## 4. Autonomous Self-Learning (Online RL)

While farming with `deploy_mode: "target"`, the bot automatically learns from every wave and every raid:
- **Between waves**: checks which enemy storages were destroyed (`+1.0` reward) vs which survived (`-0.5` penalty).
- **After each raid**: updates `models/policy_net.pt` and `models/coc_commander.pt` on your GPU and logs the learning step in the Live Log (`[Self-Learning PolicyNet] REINFORCE SUKSES` or `KOREKSI KESALAHAN`) as well as `dataset/self_learned_raids.jsonl`.
