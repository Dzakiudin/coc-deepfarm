import json
import logging
import os

from core.utils import resource_path, user_path

logger = logging.getLogger(__name__)

SETTINGS_PATH = user_path("settings.json")
DEFAULT_SETTINGS_RESOURCE = "settings.example.json"

# Universal 16:9 / Landscape Clash of Clans normalized button coordinates (0.0 .. 1.0).
# Works out-of-the-box on standard emulator resolutions (1280x720, 1600x900, 1920x1080, 1680x720)
# without requiring manual 16-button calibration.
DEFAULT_CONTROLS = {
    "find_btn": {"x": 0.12976, "y": 0.74028},
    "attack_btn": {"x": 0.7994, "y": 0.9},
    "nxt_btn": {"x": 0.93214, "y": 0.70972},
    "s_goblin": {"x": 0.20179, "y": 0.90417},
    "troop_slot_1": {"x": 0.27143, "y": 0.88889},
    "troop_slot_2": {"x": 0.32738, "y": 0.88333},
    "troop_slot_3": {"x": 0.3875, "y": 0.88889},
    "troop_slot_4": {"x": 0.43869, "y": 0.88611},
    "troop_slot_5": {"x": 0.50714, "y": 0.90694},
    "surr": {"x": 0.05417, "y": 0.74583},
    "surr_conf": {"x": 0.58155, "y": 0.63889},
    "return_home": {"x": 0.5006, "y": 0.85833},
    "retry_btn_1": {"x": 0.96845, "y": 0.075},
    "retry_btn_2": {"x": 0.70714, "y": 0.16111},
    "retry_btn_3": {"x": 0.8875, "y": 0.06667},
    "attack_btn_home": {"x": 0.0494, "y": 0.88889},
}


def ensure_default_controls(controls: dict | None) -> dict:
    """
    Return a controls dictionary guaranteed to have all 16 Clash of Clans buttons,
    preserving any custom user-calibrated coordinates while filling missing ones from DEFAULT_CONTROLS.
    """
    merged = {k: dict(v) for k, v in DEFAULT_CONTROLS.items()}
    if isinstance(controls, dict):
        for k, v in controls.items():
            if isinstance(v, dict) and "x" in v and "y" in v:
                merged[k] = dict(v)
    return merged


def _seed_default_settings():
    if os.path.exists(SETTINGS_PATH):
        return
    try:
        with open(resource_path(DEFAULT_SETTINGS_RESOURCE), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if not save_settings(data):
                logger.error("Could not write initial settings to %s", SETTINGS_PATH)
        else:
            logger.error(
                "Default settings template %s must contain a JSON object",
                DEFAULT_SETTINGS_RESOURCE,
            )
    except (OSError, ValueError) as e:
        logger.warning(
            "Could not seed settings from %s: %s", DEFAULT_SETTINGS_RESOURCE, e
        )


def load_settings():
    _seed_default_settings()
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except (OSError, ValueError) as e:
        logger.warning("Could not load settings from %s: %s", SETTINGS_PATH, e)
        pass
    return {}


def save_settings(data):
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except OSError as e:
        logger.error("Could not save settings to %s: %s", SETTINGS_PATH, e)
        return False
