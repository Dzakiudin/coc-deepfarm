# ⚙️ Configuration Reference

**Coc-DeepFarm** uses two clean configuration layers:
1. **[`core/config.py`](../core/config.py)** — Code-level `AppConfig` dataclass (model paths, ADB defaults, YOLO class labels).
2. **[`settings.json`](../settings.json)** — Runtime settings (auto-seeded from [`settings.example.json`](../settings.example.json) with the **Universal 16:9 CoC Control Preset**).

---

## 1. `AppConfig` ([`core/config.py`](../core/config.py))

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_path` | `str` | `"models/best.pt"` | Path to the Ultralytics YOLO weights for storage/collector/UI detection |
| `conf` | `float` | `0.5` | Minimum YOLO detection confidence threshold |
| `gpu` | `bool \| None` | `None` | `None` = auto-detect CUDA GPU; set `True` or `False` to force |
| `max_retries` | `int` | `4` | Scan attempts per phase before triggering recovery |
| `adb_host` | `str` | `"127.0.0.1"` | Localhost address for the emulator ADB server |
| `adb_port` | `int` | `16384` | Default port (auto-scans `5555`, `16384`, `7555`, etc. across all emulators) |
| `adb_path` | `str` | `""` | Custom `adb.exe` path (empty = auto-discover from LDPlayer, MuMu, BlueStacks, MEmu, or PATH) |

---

## 2. `settings.json` Structure

> [!NOTE]
> All 16 button coordinates in `controls` are pre-populated automatically via `DEFAULT_CONTROLS` in [`core/settings.py`](../core/settings.py). You never need to edit them manually on standard 16:9 landscape resolutions.

```json
{
  "controls": {
    "attack_btn_home": { "x": 0.0494,  "y": 0.88889 },
    "find_btn":        { "x": 0.12976, "y": 0.74028 },
    "nxt_btn":         { "x": 0.93214, "y": 0.70972 },
    "attack_btn":      { "x": 0.7994,  "y": 0.90000 },
    "s_goblin":        { "x": 0.20179, "y": 0.90417 },
    "troop_slot_1":    { "x": 0.27143, "y": 0.88889 },
    "troop_slot_2":    { "x": 0.32738, "y": 0.88333 },
    "troop_slot_3":    { "x": 0.38750, "y": 0.88889 },
    "troop_slot_4":    { "x": 0.43869, "y": 0.88611 },
    "troop_slot_5":    { "x": 0.50714, "y": 0.90694 },
    "surr":            { "x": 0.05417, "y": 0.74583 },
    "surr_conf":       { "x": 0.58155, "y": 0.63889 },
    "return_home":     { "x": 0.50060, "y": 0.85833 },
    "retry_btn_1":     { "x": 0.96845, "y": 0.07500 },
    "retry_btn_2":     { "x": 0.70714, "y": 0.16111 },
    "retry_btn_3":     { "x": 0.88750, "y": 0.06667 }
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

### Key Settings Summary

| Section & Key | Allowed Values | Purpose |
| :--- | :--- | :--- |
| `controls.*` | `{"x": 0..1, "y": 0..1}` | Normalized screen coordinates for all 16 game buttons |
| `troops.deploy_mode` | `"target"` \| `"tap"` \| `"hold"` | `"target"` uses `AttackPolicyNet` perimeter tap predictions around storages |
| `troops.skip_empty_slots` | `true` \| `false` | Uses HSV color saturation to skip depleted (grayscale) troop slots in Wave 2+ |
| `attack.target_mode` | `"smart"` \| `"and"` \| `"or"` | `"smart"` enables Combo Gold+Elixir, 1.6× Jackpot, and 15% Dead-Base threshold discount |
| `attack.drain_ratio` | `0.1 .. 1.0` (`0.8`) | Fraction of detected resource buildings to destroy before surrendering early |
| `brain.name` | `"coc_ai"` \| `"rule"` | `"coc_ai"` activates `CoCCommanderNet` + `BaseMemoryBank` on GPU |
| `notifications.enabled` | `true` \| `false` | Enables optional `ntfy.sh` mobile push notifications |
