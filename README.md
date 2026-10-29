# Coc-DeepFarm

<p align="center">
  <strong>Python-powered Clash of Clans farming automation for Android emulators</strong><br>
  Computer vision · OCR · ADB automation · Strategy planning · Raid replay
</p>

<p align="center">
  <a href="#features">Features</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#getting-started">Getting started</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#project-structure">Project structure</a>
</p>

---

## Overview

**Coc-DeepFarm** is a modular Python desktop application that observes Clash of Clans through an Android emulator, reads available loot, evaluates bases, deploys troops, and records raid outcomes.

The app uses Android Debug Bridge (ADB) for emulator frames and input, so it does not depend on desktop mouse clicks or keeping the emulator window focused.

The perception pipeline combines a trained **YOLO** detector with **EasyOCR**. Farming decisions are orchestrated by Python logic. When configured, the optional **Gemini** planner can return a structured strategy; `bot.py` validates that plan and may use it for the live raid. If the planner is unavailable or returns no valid plan, the existing Python logic remains the fallback.

## Features

- **ADB-first automation** — stream emulator frames and send taps, swipes, holds, and key events through ADB.
- **Low-latency video capture** — decode the emulator's H.264 screen stream for live frames, with standard ADB screenshots as a fallback.
- **Vision + OCR** — locate interface elements and resource storage buildings, then read resource values.
- **Configurable deployment** — choose a tap sweep, long press, or geometry-aware target deployment using detected storage locations.
- **Calibrated navigation** — send ADB taps using the saved button coordinates and configured waits.
- **Scan recovery** — use the configured surrender or retry-button coordinates after a scan failure.
- **Loot validation** — apply sanity checks to OCR-derived values before adding results to session totals.
- **Raid replay records** — write state, plan, timestamped events, and outcomes to JSON for later inspection.
- **Optional strategy comparison** — compare the existing rule-based baseline with a Gemini-generated plan.
- **Live monitoring dashboard** — emulator preview, current screen state, detected YOLO classes/confidence, recent replay records, session totals, activity logs, and measured capture/inference/OCR latency.
- **Live stream preview** — displays the latest decoded emulator frame at up to 20 FPS while farming or idle, without adding screenshot requests or queueing stale frames.
- **Desktop controls** — Tkinter interface for thresholds, goals, status, totals, and logs; **F12** is the emergency stop hotkey.
- **Build-aware paths** — support running from source or from a PyInstaller bundle.

See [the architecture guide](docs/ARCHITECTURE.md) for runtime data flow,
capture lifecycle, module responsibilities, and thread boundaries.

## Architecture

<p align="center">
  <img src="docs/Arsitektur%20Sistem%20AI%20Coc-DeepFarm.png" alt="Coc-DeepFarm system architecture" width="100%">
</p>

The application follows a **modular-monolith** design: modules have distinct responsibilities but run as one desktop application.

| Layer | Modules | Responsibility |
|---|---|---|
| Startup / composition | `main.py` | Load configuration, connect ADB, instantiate components, launch GUI |
| Orchestration | `bot.py` | Coordinate the farm loop, scanning, attack execution, recovery, and totals |
| Perception | `vision.py`, `screen.py` | YOLO detections, OCR, standalone screen-state classifier |
| Strategy | `brain.py` | Rule-based baseline and optional Gemini strategy proposal |
| Input execution | `automation.py` | ADB tap/hold helpers; farming navigation uses calibrated coordinates |
| Device transport | `adb.py`, `window_manager.py` | Emulator connection, H.264 frames, screenshot fallback, input |
| UI | `gui.py`, `calibrate.py` | Operator controls and button-coordinate calibration |
| Observability | `logger.py`, `notify.py` | File/console logs and optional ntfy notifications |
| Replay analysis | `replay.py`, `replay_report.py` | Raid JSON recording and summary reports |
| Configuration / runtime | `config.py`, `settings.py`, `utils.py` | Static defaults, machine settings, source/frozen resource paths |

### Runtime flow

1. **Initialize** — load local settings, configure logging, connect to the emulator, and start H.264 capture when FFmpeg is available.
2. **Observe** — read the latest video frame (or an ADB screenshot fallback) and run the vision/OCR pipeline.
3. **Classify and scan** — determine the current game state and read available loot and storage locations.
4. **Evaluate** — compare loot against configured thresholds and prepare a raid state.
5. **Plan** — select a deployment approach using the current Python logic; optional brain plans can be recorded for comparison.
6. **Act** — send the calibrated inputs through ADB.
7. **Recover when necessary** — retry recoverable scans using the configured surrender or retry-button coordinates.
8. **Evaluate the result** — read the post-battle summary and validate the loot values.
9. **Record and repeat** — update session totals, write replay data, update the GUI, and continue until stopped or goals are reached.

### Threading model

The Tkinter GUI runs on the main thread. Farming work runs separately in the bot worker thread, while the video decoder and preview reader run in background threads. A thread-safe `queue.Queue` carries log, status, alert, and totals messages back to the GUI; preview frames use a single latest-frame slot so a slow UI never builds a backlog of stale images. Tkinter renders at up to 30 FPS; the Android stream supplies up to 20 FPS.

### Video capture

When FFmpeg is available on `PATH` and the Android device supports raw H.264
screen recording, the app starts a video stream and uses its latest decoded
frame for the preview and vision scans. The stream is stopped when the app
closes. If FFmpeg or device support is unavailable, the app logs a warning and
falls back to one-shot ADB screenshots. The Performance panel's **Capture
Delay** metric represents the age of the latest decoded stream frame while
streaming, or the duration of a screenshot fallback.

### Strategy and Gemini

The `Brain` interface separates strategy planning from perception and input execution:

- **`RuleBrain`** provides a deterministic baseline plan.
- **`GeminiBrain`** can request a structured plan containing `skip_base`, `deploy_order`, `surrender_after_s`, and `target_side`.
- When configured and a valid response is returned, `bot.py` validates and applies the alternative plan for that raid. If the API is unavailable, the response is invalid, or no plan is returned, the bot falls back to its existing rule-based behavior.

Perception remains responsible for object positions; the planner does not return pixel coordinates. Review the configured mode and API key before running the bot.

## Getting started

### Requirements

- Windows
- Python 3.10+
- An Android emulator with ADB enabled
- A working `adb.exe`, discoverable through `PATH`, Android platform-tools, or the emulator installation
- FFmpeg available on `PATH` for live video capture; without it, the app uses slower ADB screenshots
- The trained YOLO model at `models/best.pt` (or a different path configured in `config.py`)
- A GPU is recommended for faster YOLO inference; CPU execution may also work

The base Python dependencies are listed in `requirements.txt`, including
Ultralytics, OpenCV, NumPy, EasyOCR, Pillow, and keyboard. The optional
`requirements-gpu.txt` profile installs the tested CUDA 12.8 PyTorch build for
Windows and Python 3.12.

### 1. Create an environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Optional GPU setup and FFmpeg

For an NVIDIA GPU with Python 3.12, install the optional CUDA build into the
same virtual environment:

```powershell
python -m pip install -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Install FFmpeg for Windows and add the directory containing `ffmpeg.exe` to
`PATH`, then restart the terminal or app. Video capture falls back to ADB
screenshots if FFmpeg is not installed. See
[FFmpeg downloads](https://ffmpeg.org/download.html) for Windows build options.

### 3. Connect the emulator

Enable the emulator's ADB connection and confirm that it is reachable. The
default configuration targets `127.0.0.1:16384` (MuMu Player); if that port is
unavailable, ADB probes common MuMu, LDPlayer, and MEmu ports.

### 4. Calibrate controls

Open the village screen in the emulator and run:

```powershell
python calibrate.py
```

Follow the calibration prompts and click the requested game controls. The calibration values are stored in the local `settings.json` file as guest-resolution-relative positions. Recalibrate if a game update changes button locations or the UI layout.

### 5. Start the application

```powershell
python main.py
```

Alternatively, use `run.bat` if available in your checkout. Use the GUI to configure loot thresholds and session goals. Press **F12** or use the Stop button to request a clean stop.

### 6. Check the installation

```powershell
python main.py --selftest
```

The self-test checks dependencies, reports CUDA availability, and loads the
configured YOLO model. It does not require an active emulator session.

## Configuration

| File / setting | Purpose |
|---|---|
| `config.py` / `AppConfig` | Model path, confidence threshold, GPU preference, ADB host/port, detection class names |
| `settings.json` | Machine-specific controls and runtime options; local settings should not be committed |
| `controls` | Calibrated positions for game buttons |
| `waits` | Scan, action-settle, and verification timing |
| `recovery` | Recovery limits for repeated scan failures |
| `attack` | Drain ratio and attack time limit |
| `troops` | Troop slots, deployment mode, hold timing |
| `replay` | Raid recording options and output directory |
| `brain` | Optional strategy planner configuration |
| `notifications` | Optional ntfy server, topic, and token |

`settings.json` is created from the tracked `settings.example.json` template
on first launch when no local settings file exists. It is intentionally ignored
by Git because it contains machine-specific calibrations and may contain local
service credentials. Run `calibrate.py` to fill in the required button
coordinates.

### Deployment modes

- **`tap`** — tap-sweep deployment; current default.
- **`hold`** — long-press deployment; behavior can vary by emulator.
- **`target`** — geometry-aware target deployment around detected stores.

Review the defaults in the source and your local `settings.json` before changing automation parameters.

## Project structure

| Path | Responsibility |
|---|---|
| `main.py` | Application entry point, initialization, self-test |
| `bot.py` | Main farm loop, loot/base scans, attack orchestration, recovery |
| `vision.py` | YOLO inference, detection details, OCR parsing |
| `screen.py` | Read-only game screen classification |
| `automation.py` | ADB input abstractions and optional click-verification helpers |
| `adb.py` | ADB connection, guest input, H.264 stream decoding |
| `window_manager.py` | Live video capture with ADB screenshot fallback |
| `brain.py` | Rule planner and optional Gemini planner |
| `gui.py` | Tkinter controls, status, totals, log display, live stream preview |
| `calibrate.py` | Interactive control calibration |
| `pinpoint.py` | Screenshot/pixel locator utility |
| `config.py` | Static application defaults and detection labels |
| `settings.py` | Read/write local runtime settings |
| `settings.example.json` | Safe starter settings copied on first launch |
| `tests/` | Standard-library regression tests |
| `replay.py` | Per-raid JSON recorder |
| `replay_report.py` | Replay summary and strategy comparison |
| `logger.py` | File and console logging |
| `notify.py` | Optional asynchronous ntfy notifications |
| `utils.py` | Resource paths and source/frozen runtime helpers |
| `docs/ARCHITECTURE.md` | Runtime data flow, capture lifecycle, and threading model |

## Logs and replay data

- The application log is written to `logs/bot.log`.
- When replay recording is enabled, each raid can produce a JSON file containing the scanned state, strategy plan, timestamped events, and outcome.
- Run `replay_report.py` to inspect recorded raid summaries and compare strategy plans.

Treat logs and replay files as diagnostic data. Review them before sharing because they may contain machine-specific paths or gameplay/session details.

## Troubleshooting

- **ADB connection fails:** start the emulator, enable its ADB option, and check the configured host/port.
- **The detector model cannot be loaded:** confirm that `models/best.pt` exists or update `model_path`.
- **Buttons are missed:** rerun `calibrate.py` and verify the emulator resolution.
- **Loot values look incorrect:** inspect the logs and replay data; OCR and object detection are probabilistic and can misread UI elements.
- **Gemini does not return a plan:** confirm the API key/configuration and check the logs for network, quota, or parsing errors. The live bot should not be assumed to follow a shadow-mode proposal.

## Limitations and responsible use

- This project is designed around one configured emulator session and the current farming strategy; it is not a general-purpose game automation framework.
- Computer vision and OCR can fail when game UI, resolution, language, or model assumptions change.
- Emulator behavior can differ, especially for long-press input and timing.
- Before redistributing a public build, verify that you have permission to publish the trained weights in `models/best.pt`.
- **Automation may violate Supercell's Terms of Service and can put an account at risk.** Use responsibly and understand the consequences before running it.

## License

MIT — see [LICENSE](LICENSE). This fork is based on [Coc-Auto-Farm](https://github.com/anugrhaswi/Coc-Auto-Farm); refer to the upstream project and included license for attribution and terms.
