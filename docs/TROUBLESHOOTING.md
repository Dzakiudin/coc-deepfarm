# Troubleshooting

## "Could not connect to the emulator via ADB"

Check, in order:

1. The emulator is running; its ADB port only listens while an instance is up.
2. Enable the emulator's ADB connection (for LDPlayer: **Settings → Others →
   ADB local connection**).
3. The host adb binary is findable. Auto-detection covers:
   - `adb` on `PATH` / `platform-tools`,
   - `$ANDROID_HOME\platform-tools\adb.exe`,
   - supported emulator install folders.
   Fix: run `adb connect 127.0.0.1:16384` for the default MuMu port (or your
   emulator's configured port), or set
   `config.adb_path` explicitly.
4. Confirm the configured port matches the emulator. Defaults and common port
   ranges are listed in [Configuration](CONFIGURATION.md#adb-connection-notes).
5. `adb devices` shows the serial as **`device`** (not `offline`). If `offline`,
   restart the adb server: `adb kill-server` then connect again.

## "Run calibrate.py first" when pressing Start

`settings.json` has no `controls` entry for one of the 11 required buttons. Fix:

```
python calibrate.py
```

Set all 11 required buttons, Save, relaunch `main.py`.

## "Could not find [class] after X attempts"

A single scan gave up: YOLO saw no storehouses or gold text within
`max_retries` captures. Recovery uses calibrated `surr` or `retry_btn_1/2/3`
coordinates without checking the screen or verifying each tap. This restores
the earlier navigation behavior, but can tap unrelated controls if the game is
on an unexpected screen. Each failure also pushes a notification with a
screenshot of the failing screen (saved under `logs/error_*.jpg`). If this
happens `recovery.scan_restart_limit` times **in a row** (default 3) the bot
stops. To make it more tolerant:

- Raise `recovery.scan_restart_limit` in `settings.json`.
- Slow the scans down with `waits.scan_base_first` / `scan_treasures_first`.
- Lower `config.conf` (e.g. 0.4) if boxes exist but are being thresholded out.
- Check `logs/bot.log` for what phase it was in.
- Confirm the model `models/best.pt` includes that class name (see labels in
  `config.py`).

## "No loot read for N raids in a row" (bot stopped)

The bot finished `recovery.scan_restart_limit` raids **in a row** without reading
any post-battle Gold/Elixir/Dark. That almost always means the surrender didn't
take effect (the screen never reached the loot summary), so the raid counted 0
and the bot may be desynced. Check:

- The `surr` / `surr_conf` calibration — re-run `calibrate.py` and use
  **Recapture** while actually in a battle so you click the real buttons.
- `waits.surr` / `waits.surr_conf` — if your quit dialog is slow, raise them.
- A raid where **no troops were actually deployed** also has no loot summary
  (surrendering without attacking goes straight home). If the log shows
  `providers at scan: 0` / `wave 1: sweeping 0 row(s)` the layout scan found no
  storehouse rows to sweep, so the base was never attacked.
- `logs/bot.log`: each raid logs `loot this raid: Gold +… | Elixir +… | Dark +…`,
  and unreadable post-battle reads log `post-battle loot not readable`.

The watchdog exists so this can't run silently all night; raise
`recovery.scan_restart_limit` if you'd rather tolerate more empty raids.

## "Failed to load YOLO model"

- Does `models/best.pt` exist? (Default `config.model_path` is
  `models/best.pt`, relative to the script.)
- Is it a valid Ultralytics `.pt`? First time, Ultralytics downloads any
  boilerplate — an offline/airgapped machine will fail.

## "AI mode unavailable"

The AI strategy toggle is disabled until a Gemini brain is configured and an API
key is available through `brain.api_key` or the environment variable named by
`brain.api_key_env` (default `GEMINI_API_KEY`). The rule-based mode does not need
an API key.

## Buttons work but nothing happens in the game

Background ADB taps are injected into the guest, not the desktop, so the
taps are fine even when the window is hidden. If the game state is wrong:

1. Redo calibration with the correct screens visible (see
   [Usage](USAGE.md#calibrate-buttons-calibratepy)).
2. Confirm the guest resolution the bot uses matches what you calibrated on —
   `adb -s 127.0.0.1:16384 shell wm size` (replace the port with the connected
   device's port). Normalized coords adapt automatically,
   but a *changed* resolution between calibration and farming is fine (that's
   the point); a *calibration done on the wrong screen* is not.

## No stored window / picker is gone

There is no window picker anymore — the emulator is identified purely by its
ADB port (`config.adb_port`, default `127.0.0.1:16384`). If the bot connects to
a port that isn't your emulator (e.g. a leftover from another instance),
`adb devices` will show connected devices; set `config.adb_port` to your
instance's port (MuMu `16384 + 32*N`, LDPlayer odd ports from 5555, or MEmu
`21503 + 32*N`).

## High memory usage

YOLO + EasyOCR are heavy. Close other apps, and leave `config.gpu = None` to
auto-detect CUDA. `run.bat` uses the project's `.venv`; installing CUDA PyTorch
in a different Python environment will not enable the bot's GPU.

To enable an NVIDIA GPU on Windows, install the optional CUDA 12.8 PyTorch
build into the same environment after the regular requirements:

```
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

The app's log reports the selected vision device when the farm starts.
`python main.py --selftest` also reports whether this Python build can access
CUDA. The RTX 4060 is supported by the CUDA 12.8 build; a CPU-only PyTorch
installation remains supported and runs with CPU inference.

## Screenshot capture still slow

When FFmpeg is installed and the Android device supports H.264 streaming,
startup logs should include `Live H.264 video capture started`. The GUI preview
uses the latest stream frame at up to 20 FPS; the Tkinter render loop polls at
up to 30 FPS and drops stale intermediate frames. The Performance panel's
Capture Delay measures the age of the most recently decoded video frame, not
the time for a new screenshot command. If the stream is unsupported or stops
updating, the app warns and falls back to ADB screenshots.

FFmpeg must be available as `ffmpeg.exe` on `PATH` before starting the app. If
the app reports that FFmpeg was not found, install FFmpeg for Windows, add its
`bin` directory to `PATH`, and restart the app.

The stream was tested with MuMu Android 13 and 1680×720 output. Other emulator
versions may not support raw H.264 `screenrecord`; in that case the ADB
screenshot fallback keeps capture working but the preview will be less smooth.

## F12 hotkey not registered

If the hotkey fails (permission or locked keyboard on the host), the warning is
logged. The GUI stays visible while farming, so the **Stop Farm** button is
always available as a fallback. To restore F12, try running the terminal as
Administrator so the `keyboard` library can register global hotkeys.