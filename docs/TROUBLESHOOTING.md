# Troubleshooting

## "Could not connect to the emulator via ADB"

1. Ensure the emulator is running and ADB is enabled (e.g., MuMu Player port `16384` or LDPlayer port `5555`).
2. Verify with `adb devices` that the instance shows as `device` (not `offline`).

## "Run calibrate.py first" when pressing Start

`settings.json` is missing one of the required button coordinates. Run:

```powershell
python ui/calibrate.py
```

Set all required buttons, click **Save**, and relaunch `python main.py`.

## "Could not find [class] after X attempts"

YOLO did not detect storages or resource counters within `max_retries` captures:
- Raise `recovery.scan_restart_limit` in `settings.json`.
- Increase `waits.scan_base_first` or `waits.scan_treasures_first` if your emulator loads slowly.

## Post-battle loot reads 0 or surrender dialog doesn't close

- Critical progression buttons (`surr_conf`, `return_home`, `nxt_btn`) automatically fall back to calibrated coordinates even when `click_verify_strict` is enabled.
- Re-verify `surr` and `surr_conf` in `python ui/calibrate.py` while in an active battle if the quit confirmation button moved.

## GPU / CUDA Acceleration

`main.py` automatically re-launches inside `.venv\Scripts\python.exe`. Verify CUDA availability at any time with:

```powershell
python main.py --selftest
```
