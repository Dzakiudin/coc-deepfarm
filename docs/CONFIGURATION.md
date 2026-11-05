# Configuration

There are two configuration layers:

- **`core/config.py`** — code-level `AppConfig` dataclass (model path, ADB defaults, YOLO class labels).
- **`settings.json`** — user data written at runtime (calibrated button coordinates, loot filter, troop slots, and AI brain mode).

## `AppConfig` (`core/config.py`)

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `model_path` | `str` | `"models/best.pt"` | Path to the YOLO model used for layout / treasures / loot |
| `conf` | `float` | `0.5` | YOLO confidence threshold |
| `gpu` | `bool \| None` | `None` | `None` = auto (CUDA when available); force `True`/`False` |
| `max_retries` | `int` | `4` | Attempts per vision scan before giving up |
| `adb_host` | `str` | `"127.0.0.1"` | Host the emulator ADB server listens on |
| `adb_port` | `int` | `16384` | MuMu Player instance 0; other common emulator ports are auto-scanned if unavailable |
| `adb_path` | `str` | `""` | Explicit `adb.exe`. Empty = auto-detect from PATH, Android platform-tools, or emulator folders |

## `settings.json`

Created automatically from `settings.example.json` on first launch.

```json
{
  "controls": {
    "attack_btn_home": { "x": 0.05, "y": 0.89 },
    "find_btn":        { "x": 0.13, "y": 0.74 },
    "nxt_btn":         { "x": 0.93, "y": 0.71 },
    "attack_btn":      { "x": 0.80, "y": 0.90 },
    "s_goblin":        { "x": 0.20, "y": 0.90 },
    "surr":            { "x": 0.05, "y": 0.75 },
    "surr_conf":       { "x": 0.58, "y": 0.64 },
    "return_home":     { "x": 0.50, "y": 0.86 }
  },
  "troops": {
    "slots": ["s_goblin", "troop_slot_1", "troop_slot_2", "troop_slot_3", "troop_slot_4", "troop_slot_5"],
    "deploy_mode": "target",
    "skip_empty_slots": true
  },
  "attack": {
    "drain_ratio": 0.8,
    "time_limit": 90.0,
    "target_mode": "smart"
  },
  "brain": {
    "name": "coc_ai",
    "use_ai": true
  }
}
```

| Section | Key | Purpose |
|---|---|---|
| `controls` | button names | Normalized `(x, y)` fractions written by `python ui/calibrate.py` |
| `troops.deploy_mode` | `"target"` \| `"tap"` \| `"hold"` | `"target"` uses `AttackPolicyNet` perimeter tap predictions around storages |
| `troops.skip_empty_slots` | `true` \| `false` | Measures HSV color saturation to skip depleted (grayscale) troop slots |
| `attack.target_mode` | `"smart"` \| `"and"` \| `"or"` | Loot evaluation filter (`smart` enables Combo Gold+Elixir, Jackpot, and Dead-Base boost) |
| `brain.name` | `"coc_ai"` \| `"rule"` | `"coc_ai"` runs `CoCCommanderNet` on GPU; `"rule"` uses deterministic heuristics |
| `brain.use_ai` | `true` \| `false` | Master switch for `CoCTacticalBrain` |
| `waits` | per-step floats | Optional pause/grace overrides and smart-click visual verification settings |
| `recovery.scan_restart_limit` | `int` (default `3`) | Consecutive failed scans or 0-loot raids tolerated before stopping |
| `notifications` | `enabled`, `ntfy_topic`, ... | Optional `ntfy.sh` push alerts with error/completion screenshots |
