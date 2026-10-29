# Architecture

## Overview

COC Auto Farm observes the emulator, decides when to attack, and injects taps
without requiring the emulator window to be focused or visible.

Input goes through **ADB** (`adb shell input tap`). Live capture uses an H.264
stream from Android `screenrecord`, decoded by FFmpeg. A one-shot ADB screenshot
is used to establish the framebuffer dimensions and remains the fallback if
streaming is unavailable. Fixed menu buttons use **calibrated normalized
coordinates** stored in local `settings.json`. YOLO + EasyOCR locate changing
base layouts and read resource values.

## Runtime data flow

```
                                  ┌─ ADB exec-out screenrecord ─ FFmpeg
                                  │                                 │
main.py ── WindowManager ─────────┴── VideoCaptureStream ────────────┤
    │                             (latest decoded frame)             │
    ├── CocBot worker ── capture_window() ──────────────────────────┤
    │       ├── VisionManager (YOLO / EasyOCR)                       │
    │       └── AutomationController ── ADB input                    │
    │                                                                v
    └── BotGUI ── preview reader ── latest-frame slot ── Tk render loop
              └── event queue <── CocBot worker

If video capture is unavailable or stale, WindowManager falls back to
ADB exec-out screencap -p.
```

Components and their neighbours:

| Component | Knows about | Depends on |
|-----------|-------------|------------|
| `main.py` | everything | all modules |
| `config.py` | nothing (data only) | — |
| `window_manager.py` | guest resolution, live frames, screenshot fallback | `adb.py` |
| `adb.py` | ADB commands, H.264 decoder process, latest decoded frame | subprocess, cv2, FFmpeg |
| `automation.py` | tap/key target | `adb.py` |
| `vision.py` | YOLO/EasyOCR inference | torch, ultralytics, easyocr |
| `bot.py` | the game flow | everything above |
| `gui.py` | Tkinter, event queue, latest-frame preview slot | `window_manager.py` |
| `calibrate.py` | screenshot + click-to-save | `adb.py`, settings.py, Pillow |

## Thread model

- **Main thread**: `main()` — ADB connect/gate, Tkinter GUI (`gui.py`), message
  queue polling (every 150ms), F12 hotkey.
- **Worker thread**: `bot.py` `CocBot._farm_loop()` — capture, inference, taps.
  Reads `_stop_event` in every sleep chunk, so stop is near-instant.
- **Video decoder thread**: `VideoCaptureStream` — reads FFmpeg's raw BGR frames
  from the ADB H.264 stream and retains only the newest frame.
- **Preview reader thread**: `BotGUI._idle_preview_loop()` — polls the live
  stream while farming or idle and writes the newest frame to a single
  lock-protected slot. It does not enqueue images or touch Tk widgets.
- **Preview render loop**: a Tk `after()` callback consumes the latest slot and
  renders at up to 30 FPS. If rendering falls behind, intermediate frames are
  discarded instead of accumulating a stale queue.
- **Communication**: the worker never touches Tk directly. It puts
  `{"type": "log" | "alert" | "status" | "totals", ...}` dicts on a
  `queue.Queue`; `BotGUI._poll_queue()` drains it on the main thread.

## Capture lifecycle

1. Startup makes one ADB screenshot to establish the actual framebuffer size.
2. `WindowManager.start_video_capture()` launches
   `adb exec-out screenrecord --output-format=h264` and pipes it to FFmpeg.
3. The decoder publishes complete BGR frames; the bot and GUI read copies of
   the latest frame without requesting a new screenshot for each read.
4. If FFmpeg is unavailable, the stream fails to start, or frames become stale,
   the bot capture API falls back to `adb exec-out screencap -p`. A stream that
   ends is restarted in the background.
5. On shutdown, the stream processes and decoder thread are stopped.

The GUI's **Capture Delay** metric is the age of the latest decoded stream
frame while streaming. In screenshot fallback mode it is the elapsed time for
the ADB screenshot operation.

## Coordinate systems

Two coordinate spaces exist; the layers keep them straight:

| Space | Meaning | Producer |
|-------|---------|----------|
| **Guest pixels** | Pixels inside the emulator's Android screen | H.264 frames, `adb.screencap()`, vision detections |
| **Normalized** | Fractions `(0..1, 0..1)` of the guest screen | `calibrate.py` → `settings.json` |

`WindowManager.win_left = win_top = 0`, `win_width/win_height` = guest
resolution. `_press_button` computes `pixel = fraction * guest_dimension` and
taps via ADB. Vision returns frame-local (guest) pixels, clicked directly.
No desktop offsets, no window focus — the emulator can stay minimized.

### Fixed buttons vs. per-base content

| Item | Mechanism | Why |
|------|-----------|-----|
| `attack_btn_home`, `find_btn`, `nxt_btn`, `attack_btn`, `s_goblin`, `surr`, `surr_conf`, `return_home` | calibrated normalized coords | positions don't change between bases; calibrate once |
| `gold`, `elixir`, `d_elixir`, storehouses (`golds`/`elixirs`/`d_elixirs`), post-battle labels | YOLO + OCR | positions/layout vary per base |

There is **no YOLO fallback** for fixed buttons: if a `controls` entry is
missing, `CocBot.start()` refuses to farm and tells the user to run
`calibrate.py`. This is deliberate — the whole point is that button layout
changes (game updates, resolution) never require model retraining.

## Normalized controls (in plain words)

Instead of remembering buttons as exact pixels ("left: 800, top: 630") — which
break when the game changes resolution or the emulator is resized — each button
is stored as a percentage of the screen:

```
"rules of thumb" example:
  attack_home -> x = 0.50  (halfway left->right)
                 y = 0.85  (85% of the way down)
```

`bot.py` multiplies the percentage by the current guest resolution
(`0.85 × 900 = 765` pixels down on a 900-tall screen). Same tap position on
any resolution — and when a game update moves buttons, you recalibrate by
clicking the new spots, no YOLO retraining.

## Farm loop (bot.py `_farm_loop`)

0. **Warm-up — models**: set status `loading`, load YOLO + EasyOCR, capture a
   frame, then run `vision.warmup(frame)` before the first raid. This moves the
   first inference/OCR startup cost out of the scan flow. A load failure becomes
   a normal error (log + optional notification).
1. **Phase 1 — navigate**: `_press_button("attack_btn_home")`,
   `("find_btn")`, `("attack_btn")`, ~0.5s between taps (calibrated).
2. **Phase 2 — search & decide**: `_scan_base_layout()` (YOLO storehouses) +
   `_scan_treasures()` (YOLO boxes → OCR values). If resources meet targets →
   `_initiate_attack()` (multi-wave: taps `s_goblin` and sweeps the horizontal
   rows of every gold/elixir/dark store above the calibrated `surr` Y, then
   re-scans and repeats the wave until `attack.drain_ratio` of the providers seen
   at scan time are gone, `attack.time_limit` seconds elapse, or a Stop is
   requested between waves; a started sweep always runs to completion, then
   pauses `waits.attack_wave_settle` before rescanning; after the loop it pauses
   `waits.attack_settle` before surrendering), then
   `_press_button("surr")` (opens the quit
   dialog), `_press_button("surr_conf")` (red confirm),
   `_scan_post_game()` (data-driven: polls every interval, keeps the max value
   per resource and drops duplicate-row boxes, stopping when all three read
   non-zero or the summary's Return Home is up with a readable value; else
   best-of/0 at the timeout), add to totals, `_press_button("return_home")`.
   Otherwise `_press_button("nxt_btn")` to skip and loop.
3. **Scan-failure recovery**: if a scan gives up it raises `ScanFailed` and the
   loop restarts at Phase 1. Every failure first sends a notification with a
   screenshot of the failing screen (captured before recovery). The first
   resource-scan failure taps `surr`; all other scan failures tap the calibrated
   retry spots 1, 2, and 3. These taps use saved coordinates without screen
   detection or click verification. A consecutive-failure streak counter is
   reset by any successful scan and, at
   `recovery.scan_restart_limit`, stops the session (with a notification).
4. **0-loot watchdog**: a raid whose post-battle Gold/Elixir/Dark are all 0
   (usually a missed surrender / unreadable post-battle screen) increments a
   separate streak; any raid with loot resets it. At
   `recovery.scan_restart_limit` empty raids in a row the bot notifies and stops,
   so a desync can't run silently all night.
5. Stop when totals exceed goals, F12 pressed, or an error occurs.

`_scan_base_layout` retries up to `config.max_retries` then raises `ScanFailed`;
`_scan_treasures` does the same but raises `TreasureScanFailed` (a `ScanFailed`
subclass); `_scan_post_game` polls every `scan_post_interval` up to `scan_post_timeout`,
keeping the max value per resource and dropping boxes that overlap a row already
read; it returns once all three are non-zero, or once `return_home` is visible
with a readable value (otherwise the best partial, or 0 if nothing was
readable). On F12 a `RuntimeError("Bot stopped")` unwinds the loop cleanly (it
is not a `ScanFailed`).