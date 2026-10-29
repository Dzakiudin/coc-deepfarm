print("COC Auto-Farm: starting...", flush=True)

import atexit
import logging
import os
import queue
import sys
import tkinter as tk
from tkinter import messagebox, ttk

import keyboard

from config import AppConfig
from adb import EmulatorAdb
from logger import setup_logging
from window_manager import WindowManager
from settings import load_settings
from vision import VisionManager
from automation import AutomationController
from bot import CocBot
from gui import BotGUI
from notify import Notifier
from utils import configure_runtime_env, resource_path

logger = logging.getLogger(__name__)

_root = None


def _selftest():
    configure_runtime_env()

    checks = [
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("PIL", "PIL"),
        ("keyboard", "keyboard"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("ultralytics", "ultralytics"),
        ("easyocr", "easyocr"),
    ]
    failures = []
    for label, module in checks:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            print(f"  OK   {label:<12} {version}")
            if label == "torch":
                cuda_version = getattr(mod.version, "cuda", None)
                if mod.cuda.is_available():
                    print(f"       CUDA {cuda_version}; GPU: {mod.cuda.get_device_name(0)}")
                else:
                    print(f"       CUDA unavailable (runtime: {cuda_version or 'CPU-only build'})")
        except Exception as e:
            failures.append(f"{label}: {type(e).__name__}: {e}")
            print(f"  FAIL {label:<12} {type(e).__name__}: {e}")

    try:
        from ultralytics import YOLO
        from config import AppConfig
        from utils import resource_path

        model = resource_path(AppConfig().model_path)
        if not os.path.exists(model):
            failures.append(f"YOLO model missing: {model}")
            print(f"  FAIL yolo model   not found: {model}")
        else:
            YOLO(model)
            print(f"  OK   yolo model   {model}")
    except Exception as e:
        failures.append(f"ultralytics/YOLO: {type(e).__name__}: {e}")
        print(f"  FAIL yolo model   {type(e).__name__}: {e}")

    print()
    if failures:
        print(f"SELFTEST FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SELFTEST PASSED")
    return 0


def _show_splash(root, message):
    win = tk.Toplevel(root)
    win.title("COC Auto-Farm")
    win.resizable(False, False)
    win.transient(root)

    frame = ttk.Frame(win, padding=20)
    frame.pack()
    ttk.Label(frame, text="COC Auto-Farm", font=("Segoe UI", 12, "bold")).pack(
        pady=(0, 10)
    )
    label = ttk.Label(frame, text=message)
    label.pack()

    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.update_idletasks()
    return win, label


def main():
    setup_logging()
    logger.info("COC Auto-Farm starting")

    configure_runtime_env()

    config = AppConfig()

    model_full_path = resource_path(config.model_path)
    if os.path.exists(model_full_path):
        config.model_path = model_full_path

    settings = load_settings()
    notifier = Notifier(settings.get("notifications"))

    root = tk.Tk()
    root.withdraw()
    global _root
    _root = root

    splash, splash_label = _show_splash(root, "Starting...")

    def step(text):
        splash_label.config(text=text)
        splash.update()

    adb = EmulatorAdb(
        host=config.adb_host, port=config.adb_port, adb_path=config.adb_path
    )
    step("Connecting to the emulator over ADB...")
    splash.update()
    if not adb.connect():
        logger.error(f"ADB connect failed: {adb.last_error}")
        notifier.notify(
            "COC Auto-Farm: ADB error",
            "Could not connect to the emulator via ADB. "
            f"Details: {adb.last_error}",
            blocking=True,
            priority="high",
            tags="warning",
        )
        splash.destroy()
        messagebox.showerror(
            "ADB Error",
            "Could not connect to the emulator via ADB.\n\n"
            "Start the emulator and enable 'ADB local connection' in LDPlayer "
            "settings (Settings > Others).\n"
            "Expected device: 127.0.0.1:5555 (first instance).\n\n"
            f"Details: {adb.last_error}",
        )
        sys.exit(1)

    step("Loading components...")
    splash.update()

    window = WindowManager(adb=adb)
    step("Starting live video capture...")
    splash.update()
    window.start_video_capture()
    atexit.register(window.stop_video_capture)
    vision = VisionManager(config)
    automation = AutomationController(adb=adb)
    controls = settings.get("controls") or {}
    waits = settings.get("waits") or {}
    recovery = settings.get("recovery") or {}
    attack = settings.get("attack") or {}
    troops = settings.get("troops") or {}
    replay = settings.get("replay") or {}
    brain = settings.get("brain") or {}

    msg_queue = queue.Queue()
    bot = CocBot(config, window, vision, automation, msg_queue, controls=controls, waits=waits, notifier=notifier, recovery=recovery, attack=attack, troops=troops, replay=replay, brain=brain)

    splash.destroy()
    app = BotGUI(root, bot, msg_queue, settings=settings)
    root.deiconify()

    try:
        keyboard.add_hotkey("f12", bot.stop)
    except Exception as e:
        logger.warning(f"Could not register F12 hotkey: {e}")

    try:
        root.mainloop()
    finally:
        window.stop_video_capture()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback

        detail = traceback.format_exc()
        logger.critical("Fatal error:\n%s", detail)
        try:
            if _root is None:
                _root = tk.Tk()
                _root.withdraw()
            messagebox.showerror(
                "COC Auto-Farm crashed",
                "An unexpected error occurred.\n\n"
                f"{detail[-1500:]}\n\n"
                "The full traceback was written to the log file.",
            )
        except Exception:
            pass
        sys.exit(1)
