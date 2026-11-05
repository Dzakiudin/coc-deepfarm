import os
import sys

APP_DIR_NAME = "COCAutoFarm"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> str:
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_data_dir() -> str:
    if is_frozen():
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        path = os.path.join(base, APP_DIR_NAME)
    else:
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(path, exist_ok=True)
    return path


def resource_path(relative_path: str) -> str:
    return os.path.join(bundle_dir(), relative_path)


def user_path(relative_path: str) -> str:
    path = os.path.join(user_data_dir(), relative_path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def configure_runtime_env():
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "True")
    if is_frozen():
        os.environ.setdefault("YOLO_CONFIG_DIR", user_data_dir())
