import json
import logging
import os

from utils import resource_path, user_path

logger = logging.getLogger(__name__)

SETTINGS_PATH = user_path("settings.json")
DEFAULT_SETTINGS_RESOURCE = "settings.example.json"


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
