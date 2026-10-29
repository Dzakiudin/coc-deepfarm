import inspect
import logging
import math
import os
import queue
import threading
import time

import cv2

from brain import GeminiBrain, RuleBrain
from config import AppConfig
from notify import Notifier
from replay import ReplayLogger
from utils import user_path

logger = logging.getLogger(__name__)

DEFAULT_WAITS = {
    AppConfig.ATTACK_BTN_HOME: 0.5,
    AppConfig.FIND_BTN: 0.5,
    AppConfig.NXT_BTN: 1.5,
    AppConfig.ATTACK_BTN: 0.5,
    AppConfig.SURR_BTN: 1.0,
    AppConfig.SURR_CONF: 1.0,
    AppConfig.RETURN_HOME: 0.5,
    AppConfig.RETRY_BTN_1: 0.5,
    AppConfig.RETRY_BTN_2: 0.5,
    AppConfig.RETRY_BTN_3: 0.5,
    "scan_base_first": 3.0,
    "scan_base_retry": 2.0,
    "scan_treasures_first": 1.0,
    "scan_treasures_retry": 0.5,
    "scan_post_interval": 0.3,
    "scan_post_timeout": 12.0,
    "attack_settle": 5.0,
    "attack_wave_settle": 5.0,
}

NOTIFY_IMAGE_MAX_WIDTH = 1280
NOTIFY_IMAGE_QUALITY = 75

_BOX_OVERLAP_IOU = 0.6

_STORAGES_MAX = 16
_HISTORY_MAX = 5

_SIDE_X_BOUNDS = {"all": (0.0, 1.0), "left": (0.0, 0.5), "right": (0.5, 1.0)}
_SIDE_Y_BOUNDS = {"all": (0.0, 1.0), "top": (0.0, 0.5), "bottom": (0.5, 1.0)}

_MAX_BRAIN_SKIP_STREAK = 5

_MIN_SURRENDER_S = 10.0
_MAX_SURRENDER_S = 180.0


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


DEFAULT_RECOVERY = {
    "scan_restart_limit": 3,
}

DEFAULT_ATTACK = {
    "drain_ratio": 0.8,
    "time_limit": 30.0,
}

DEFAULT_TROOPS = {
    "slots": [AppConfig.S_GOBLIN],
    "deploy_mode": "tap",
    "hold_seconds": 0.6,
    "max_hold_points": 6,
}

_SWEEP_STEP = 50
_SWEEP_RIGHT_MARGIN = 100

_HOLD_LEFT_INSET = 0.08

_ATTACK_ROW_MARGIN = 100

_ATTACK_ROW_TOLERANCE = 20

_TARGET_RING_RADIUS = 70
_TARGET_RINGS = (70, 115)
_TARGETS_PER_STORE = 4
_TARGET_SPREAD = 10
_TARGETS_MAX = 64

_LOOT_CLAMP_SLACK = 1.10


class ScanFailed(RuntimeError):
    pass


class TreasureScanFailed(ScanFailed):
    pass


class CocBot:

    def __init__(
        self,
        config: AppConfig,
        window_manager,
        vision_manager,
        automation_controller,
        msg_queue: queue.Queue,
        controls: dict | None = None,
        waits: dict | None = None,
        notifier: Notifier | None = None,
        recovery: dict | None = None,
        attack: dict | None = None,
        troops: dict | None = None,
        replay: dict | None = None,
        brain=None,
    ):
        self.config = config
        self.window = window_manager
        self.vision = vision_manager
        self.auto = automation_controller
        self.msg_queue = msg_queue
        self.controls = controls or {}
        self.waits = {
            k: max(0.0, float(v)) for k, v in {**DEFAULT_WAITS, **(waits or {})}.items()
        }
        self.recovery = {**DEFAULT_RECOVERY, **(recovery or {})}
        self.attack = {**DEFAULT_ATTACK, **(attack or {})}
        self.troops = {**DEFAULT_TROOPS, **(troops or {})}
        if not isinstance(self.troops.get("slots"), (list, tuple)):
            self.troops["slots"] = [self.config.S_GOBLIN]
        else:
            self.troops["slots"] = list(self.troops["slots"])
        if self.troops.get("deploy_mode") not in ("tap", "hold", "target"):
            self.troops["deploy_mode"] = "tap"
        self.notifier = notifier or Notifier()
        self.replay_config = dict(replay or {})
        self.brain_config = dict(brain or {})
        self.brain = RuleBrain()
        self.brain_alt = None
        self._side_requested = ""
        try:
            name = str(self.brain_config.get("name") or "rule").strip().lower()
            if name in ("gemini", "llm"):
                self.brain_alt = GeminiBrain(
                    model=self.brain_config.get("model", "gemini-3.8-flash"),
                    api_key=self.brain_config.get("api_key") or None,
                    api_key_env=self.brain_config.get(
                        "api_key_env", "GEMINI_API_KEY"
                    ),
                    timeout_s=float(self.brain_config.get("timeout_s", 8.0)),
                    endpoint=self.brain_config.get("endpoint") or None,
                        thinking_level=self.brain_config.get("thinking_level", "low"),
                        fallback_model=self.brain_config.get("fallback_model") or None,
                    )
        except Exception as e:
            logger.warning(f"could not initialise alternate brain: {e}")
        self.ai_enabled = bool(
            self.brain_alt is not None
            and self.brain_alt.enabled
            and self.brain_config.get("use_ai", True)
        )
        self.replay = ReplayLogger(
            directory=self.replay_config.get("dir", "logs"),
            enabled=bool(self.replay_config.get("enabled", True)),
        )
        self._stop_event = threading.Event()
        self._thread = None
        self._stop_request_source = None
        self._uncalibrated_slots_logged = set()
        self._last_attack_stats = {}
        self._raid_history = []
        self._brain_skip_streak = 0
        self._current_screen = "unknown"
        self._current_stage = "Ready"
        self._last_ui_frame_at = 0.0

    def start(self, targets, goals):
        if self._thread and self._thread.is_alive():
            self._log("Bot is already running")
            return

        missing = [
            name
            for name in (
                self.config.ATTACK_BTN_HOME,
                self.config.FIND_BTN,
                self.config.NXT_BTN,
                self.config.ATTACK_BTN,
                self.config.S_GOBLIN,
                self.config.SURR_BTN,
                self.config.SURR_CONF,
                self.config.RETURN_HOME,
                self.config.RETRY_BTN_1,
                self.config.RETRY_BTN_2,
                self.config.RETRY_BTN_3,
            )
            if name not in self.controls
        ]
        if missing:
            msg = (
                "Cannot start: button positions missing for "
                + ", ".join(missing)
                + ". Run calibrate.py first."
            )
            self._log(msg)
            self._alert(msg)
            self._notify("COC Auto-Farm: cannot start", msg)
            return

        self._stop_event.clear()
        self._stop_request_source = None
        self._log("Lifecycle diagnostic build: 2026-10-27-stop-source")
        self._thread = threading.Thread(
            target=self._farm_loop,
            args=(targets, goals),
            daemon=True,
            name="CocBotFarm",
        )
        self._thread.start()
        self._set_status("running")
        self._log("Bot started")

    def stop(self):
        try:
            caller = inspect.stack()[1]
            source = f"{caller.function} ({os.path.basename(caller.filename)}:{caller.lineno})"
        except Exception:
            source = threading.current_thread().name
        self._stop_request_source = source
        self._log(f"Stop requested by {source}")
        self._stop_event.set()
        self._set_status("stopping")

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_ai_enabled(self, enabled: bool):
        if self.is_running():
            raise RuntimeError("AI mode can only be changed while the bot is stopped.")
        if enabled and (
            self.brain_alt is None or not getattr(self.brain_alt, "enabled", False)
        ):
            raise RuntimeError(
                "AI mode is unavailable. Configure a Gemini API key and restart the app."
            )
        self.ai_enabled = bool(enabled)
        mode = "AI with rule fallback" if self.ai_enabled else "rule-based"
        self._log(f"strategy set to {mode}")

    def _log(self, text: str):
        logger.info(text)
        if self.msg_queue is not None:
            self.msg_queue.put({"type": "log", "text": text})

    def _alert(self, text: str):
        if self.msg_queue is not None:
            self.msg_queue.put({"type": "alert", "text": text})

    def _set_status(self, state: str):
        if self.msg_queue is not None:
            self.msg_queue.put({"type": "status", "state": state})

    def _show_totals(self, totals: dict):
        if self.msg_queue is not None:
            self.msg_queue.put({"type": "totals", "data": totals})

    def _publish_dashboard(
        self, frame=None, details=None, *, screen=None, stage=None, loot=None, raid=None
    ):
        if self.msg_queue is None:
            return
        if screen:
            self._current_screen = str(screen)
        if stage:
            self._current_stage = str(stage)

        now_mono = time.monotonic()
        payload = {
            "type": "dashboard",
            "timestamp": time.time(),
            "perf_sample": details is not None or loot is not None,
            "screen": self._current_screen,
            "stage": self._current_stage,
            "metrics": {
                "capture_ms": float(getattr(self.window, "last_capture_ms", 0.0) or 0.0),
                "yolo_ms": float(getattr(self.vision, "last_inference_ms", 0.0) or 0.0),
                "ocr_ms": float(getattr(self.vision, "last_ocr_ms", 0.0) or 0.0),
            },
        }
        if loot is not None:
            payload["loot"] = dict(loot)
        if raid is not None:
            payload["raid"] = dict(raid)
        if details is not None:
            payload["detections"] = {
                name: [
                    {
                        "bbox": list(item.get("bbox", ())),
                        "center": list(item.get("center", ())),
                        "confidence": float(item.get("confidence", 0.0) or 0.0),
                    }
                    for item in items
                ]
                for name, items in details.items()
            }
        if (
            frame is not None
            and not getattr(self.window, "video_stream_active", False)
            and now_mono - self._last_ui_frame_at >= 0.85
        ):
            try:
                image = frame
                height, width = image.shape[:2]
                if width > 900:
                    image = cv2.resize(
                        image, (900, max(1, int(height * 900 / width))),
                        interpolation=cv2.INTER_AREA
                    )
                payload["frame"] = image.copy()
                self._last_ui_frame_at = now_mono
            except Exception as e:
                logger.debug(f"dashboard frame publish failed: {e}")
        self.msg_queue.put(payload)

    @staticmethod
    def _drain_summary(drains: list) -> str:
        if not drains:
            return "drain per raid: none completed"
        per = ", ".join(f"{d * 100:.0f}%" for d in drains)
        avg = sum(drains) / len(drains)
        return (
            f"drain per raid: {per} | "
            f"average {avg * 100:.0f}% over {len(drains)} raid(s)"
        )

    @staticmethod
    def _drain_average(drains: list) -> str:
        if not drains:
            return ""
        avg = sum(drains) / len(drains)
        return f" | avg drain {avg * 100:.0f}%"

    def _capture_screenshot(self) -> str | None:
        try:
            frame = self.window.capture_window()
        except Exception as e:
            logger.debug(f"screenshot capture failed: {e}")
            return None

        try:
            h, w = frame.shape[:2]
            if w > NOTIFY_IMAGE_MAX_WIDTH:
                scale = NOTIFY_IMAGE_MAX_WIDTH / float(w)
                frame = cv2.resize(
                    frame,
                    (NOTIFY_IMAGE_MAX_WIDTH, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            log_dir = user_path("logs")
            os.makedirs(log_dir, exist_ok=True)
            path = os.path.join(
                log_dir, "error_" + time.strftime("%Y%m%d_%H%M%S") + ".jpg"
            )
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, NOTIFY_IMAGE_QUALITY])
            return path
        except Exception as e:
            logger.debug(f"screenshot save failed: {e}")
            return None

    def _notify(self, title: str, message: str, success: bool = False):
        if not getattr(self.notifier, "is_configured", False):
            return
        try:
            image = self._capture_screenshot()
            self.notifier.notify(
                title,
                message,
                image_path=image,
                priority="default" if success else "high",
                tags="white_check_mark" if success else "warning",
            )
        except Exception as e:
            logger.debug(f"notification dispatch failed: {e}")

    def _sleep(self, seconds: float, why: str | None = None):
        started = time.monotonic()
        deadline = started + max(0.0, seconds)
        if not self._stop_event.is_set():
            self._stop_event.wait(max(0.0, deadline - time.monotonic()))
        if why is not None:
            logger.debug(
                f"waited {time.monotonic() - started:.2f}s "
                f"(target {seconds:.2f}s) for {why}"
            )

    def _button_coords(self, name: str):
        norm = self.controls.get(name)
        if not norm or "x" not in norm or "y" not in norm:
            raise KeyError(name)

        if self.window.win_width < 2 or self.window.win_height < 2:
            frame = self.window.capture_window()
            self._publish_dashboard(frame=frame)
        try:
            nx, ny = float(norm["x"]), float(norm["y"])
        except (TypeError, ValueError):
            raise RuntimeError(f"Invalid calibration for '{name}': x/y must be numeric") from None
        if not (0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0):
            raise RuntimeError(
                f"Invalid calibration for '{name}': normalized x/y must be between 0 and 1"
            )
        width, height = self.window.win_width, self.window.win_height
        if width < 2 or height < 2:
            raise RuntimeError(
                f"Cannot safely tap '{name}': invalid framebuffer size {width}x{height}"
            )
        return (
            min(width - 1, max(0, int(nx * width))),
            min(height - 1, max(0, int(ny * height))),
        )

    def _press_button(self, name: str, wait_after: float | None = None):
        try:
            x, y = self._button_coords(name)
        except KeyError:
            raise RuntimeError(f"'{name}' is not calibrated") from None
        norm = self.controls.get(name, {})
        wait = self.waits.get(name, DEFAULT_WAITS.get(name, 0.5)) if wait_after is None else wait_after
        self._log(
            f"tap {name} at ({x}, {y})"
            f" [{norm.get('x', 0):.3f}, {norm.get('y', 0):.3f}]"
            f" frame {self.window.win_width}x{self.window.win_height} wait {wait:g}s"
        )
        self.auto.click(x, y)
        self._sleep(wait)

    def _detect_scan_details(self, frame, targets):
        details = self.vision.detect_with_details(frame, targets)
        if details:
            return details

        try:
            configured_conf = float(self.config.conf)
        except (TypeError, ValueError):
            configured_conf = 0.5
        fallback_conf = min(configured_conf * 0.5, 0.25)
        if fallback_conf <= 0 or fallback_conf >= configured_conf:
            return {}

        details = self.vision.detect_with_details(
            frame, targets, conf=fallback_conf
        )
        if details:
            self._log(
                f"scan detections recovered at lower confidence "
                f"{fallback_conf:.2f}: {', '.join(sorted(details))}"
            )
        return details

    def _scan_treasures(self):
        max_retries = self.config.max_retries
        targets = [self.config.GOLD, self.config.ELIXIR, self.config.D_ELIXIR]

        for attempt in range(1, max_retries + 1):
            if self._stop_event.is_set():
                raise RuntimeError("Bot stopped")

            if attempt == 1:
                self._sleep(self.waits["scan_treasures_first"], why="first treasure scan")
            else:
                self._sleep(self.waits["scan_treasures_retry"], why="treasure scan retry")

            frame = self.window.capture_window()
            details = self._detect_scan_details(frame, targets)
            if not details:
                continue

            result = {}
            for cls in targets:
                if cls in details:
                    bbox = details[cls][0]["bbox"]
                    result[cls] = self.vision.read_number(frame, bbox)

            if result:
                logger.debug(f"base treasures read: {result}")
                self._publish_dashboard(frame=frame, details=details, screen="scouting", stage="Reading available loot", loot=result)
                return result

        raise TreasureScanFailed(
            f"Could not find treasures {targets} after {max_retries} attempts"
        )

    def _scan_base_layout(self):
        max_retries = self.config.max_retries
        targets = [
            self.config.ELIXIRS,
            self.config.GOLDS,
            self.config.D_ELIXIRS,
        ]

        for attempt in range(1, max_retries + 1):
            if self._stop_event.is_set():
                raise RuntimeError("Bot stopped")

            if attempt == 1:
                self._sleep(self.waits["scan_base_first"], why="base layout scan")
            else:
                self._sleep(self.waits["scan_base_retry"], why="base layout scan retry")

            frame = self.window.capture_window()
            details = self._detect_scan_details(frame, targets)
            if details:
                self._publish_dashboard(frame=frame, details=details, screen="scouting", stage="Scanning storage layout")
                return {
                    cls: [item["center"] for item in items]
                    for cls, items in details.items()
                }

        raise ScanFailed(
            f"Could not find base layout {targets} after {max_retries} attempts"
        )

    def _scan_post_game(self, base_loot=None):
        targets = [
            self.config.POST_GOLD,
            self.config.POST_ELIXIR,
            self.config.POST_D_ELIXIR,
        ]
        base = base_loot or {}
        caps = {}
        for cls, key in (
            (self.config.POST_GOLD, "gold"),
            (self.config.POST_ELIXIR, "elixir"),
            (self.config.POST_D_ELIXIR, "delixir"),
        ):
            try:
                offered = int(base.get(key, 0))
            except (TypeError, ValueError):
                offered = 0
            if offered > 0:
                caps[cls] = int(offered * _LOOT_CLAMP_SLACK)
        interval = self.waits["scan_post_interval"]
        deadline = time.time() + self.waits["scan_post_timeout"]
        post_conf = max(0.3, min(self.config.conf, 0.35))
        summary_marker = self.config.RETURN_HOME
        best = {}
        polls = 0

        while True:
            if self._stop_event.is_set():
                raise RuntimeError("Bot stopped")

            frame = self.window.capture_window()
            details = self.vision.detect_with_details(
                frame, targets + [summary_marker], conf=post_conf
            )
            polls += 1

            kept = []
            rejected = set()
            for cls in targets:
                for item in details.get(cls, []):
                    bbox = item["bbox"]
                    if any(_iou(bbox, seen) > _BOX_OVERLAP_IOU for seen in kept):
                        continue
                    kept.append(bbox)
                    value = int(self.vision.read_number(frame, bbox))
                    cap = caps.get(cls)
                    if cap is not None and value > cap:
                        logger.debug(
                            f"post-game {cls} read rejected: "
                            f"raw {value} > cap {cap}"
                        )
                        rejected.add(cls)
                        continue
                    if value > best.get(cls, 0):
                        best[cls] = value

            logger.debug(f"post-game poll {polls}: {best}")
            self._publish_dashboard(frame=frame, details=details, screen="post_battle", stage="Reading raid results", loot=best)
            self._dump_post_frame(frame, polls)

            if all(best.get(cls, 0) > 0 for cls in targets):
                result = {cls: str(v) for cls, v in best.items()}
                logger.debug(f"post-game loot read: {result} (after {polls} poll(s))")
                return result

            if (
                details.get(summary_marker)
                and any(v > 0 for v in best.values())
                and not rejected
            ):
                result = {cls: str(v) for cls, v in best.items()}
                logger.debug(
                    f"post-game loot read: {result} "
                    f"(summary up after {polls} poll(s))"
                )
                return result

            if time.time() >= deadline:
                if not any(v > 0 for v in best.values()):
                    self._log(
                        f"post-battle loot not readable after "
                        f"{self.waits['scan_post_timeout']:g}s; counted as 0"
                    )
                    return {}
                result = {cls: str(v) for cls, v in best.items()}
                logger.debug(
                    f"post-game loot read: {result} (partial after {polls} poll(s))"
                )
                return result

            self._sleep(interval, why="post-game loot poll")

    def _dump_post_frame(self, frame, poll):
        debug_dir = os.environ.get("COC_POST_DEBUG")
        if not debug_dir:
            return
        try:
            os.makedirs(debug_dir, exist_ok=True)
            path = os.path.join(
                debug_dir,
                "post_" + time.strftime("%Y%m%d_%H%M%S") + f"_poll{poll}.png",
            )
            cv2.imwrite(path, frame)
        except Exception as e:
            logger.debug(f"post-frame dump failed: {e}")

    def _provider_counts(self, base: dict) -> dict:
        return {
            cls: len(base.get(cls, []))
            for cls in (self.config.GOLDS, self.config.ELIXIRS, self.config.D_ELIXIRS)
        }

    def _validate_plan(self, plan: dict, state: dict) -> dict:
        if not isinstance(plan, dict) or not plan:
            return plan
        side = str(plan.get("target_side") or "all").strip().lower()
        if side not in ("all", "left", "right"):
            self._log(f"plan side {side!r} not supported; using all")
            plan = dict(plan)
            plan["target_side"] = "all"
            side = "all"
        self._side_requested = side
        if side != "all":
            self._log(f"brain asked for side {side}; deploying all sides")
            plan = dict(plan)
            plan["target_side"] = "all"
        return plan

    def _base_geometry(self, base: dict) -> dict:
        width = float(self.window.win_width or 1)
        height = float(self.window.win_height or 1)
        storages = []
        for cls, label in (
            (self.config.GOLDS, "gold"),
            (self.config.ELIXIRS, "elixir"),
            (self.config.D_ELIXIRS, "delixir"),
        ):
            for item in base.get(cls, []):
                try:
                    x, y = float(item[0]), float(item[1])
                except (TypeError, IndexError, ValueError):
                    continue
                if not (0 <= x <= width and 0 <= y <= height):
                    continue
                storages.append(
                    {"cls": label, "x": round(x / width, 3), "y": round(y / height, 3)}
                )
        storages.sort(key=lambda s: (s["y"], s["x"]))
        storages = storages[:_STORAGES_MAX]
        geometry: dict = {"storages": storages}
        if storages:
            xs = [s["x"] for s in storages]
            ys = [s["y"] for s in storages]
            geometry["bbox"] = {
                "x1": min(xs),
                "y1": min(ys),
                "x2": max(xs),
                "y2": max(ys),
            }
        return geometry

    def _build_raid_state(self, base: dict, treasure: dict, targets=None) -> dict:
        gtarget, etarget, detarget = targets or (0, 0, 0)
        history = list(self._raid_history[-_HISTORY_MAX:])
        session = {
            "raids": len(history),
            "totals": {
                key: sum(int(raid.get(key, 0)) for raid in history)
                for key in ("gold", "elixir", "delixir")
            },
        }
        return {
            "loot": {
                "gold": int(treasure.get(self.config.GOLD, 0)),
                "elixir": int(treasure.get(self.config.ELIXIR, 0)),
                "delixir": int(treasure.get(self.config.D_ELIXIR, 0)),
            },
            "targets": {
                "gold": int(gtarget),
                "elixir": int(etarget),
                "delixir": int(detarget),
            },
            "providers": self._provider_counts(base),
            "base": self._base_geometry(base),
            "history": history,
            "session": session,
            "troop_slots": self._deploy_slots(),
            "deploy_mode": self.troops.get("deploy_mode", "tap"),
            "attack": {
                "drain_ratio": float(self.attack.get("drain_ratio", 0.8)),
                "time_limit": float(self.attack.get("time_limit", 90.0)),
            },
        }

    def _count_providers_now(self) -> int | None:
        try:
            frame = self.window.capture_window()
            details = self.vision.detect_with_details(
                frame,
                [self.config.GOLDS, self.config.ELIXIRS, self.config.D_ELIXIRS],
            )
            return sum(self._provider_counts(details).values())
        except Exception as e:
            logger.debug(f"pre-surrender provider scan failed: {e}")
            return None

    def _attack_rows(self, base: dict, side: str = "all") -> list:
        _, limit = self._button_coords(self.config.SURR_BTN)
        y_lo, y_hi = _SIDE_Y_BOUNDS.get(side, _SIDE_Y_BOUNDS["all"])
        top = self.window.win_top
        height = self.window.win_height or 1
        y_min = top + int(round(height * y_lo))
        y_max = top + int(round(height * y_hi))
        rows = []
        for cls in (self.config.GOLDS, self.config.ELIXIRS, self.config.D_ELIXIRS):
            for item in base.get(cls, []):
                y = item[1]
                if y >= limit - _ATTACK_ROW_MARGIN:
                    continue
                if side in ("top", "bottom") and not (y_min <= y <= y_max):
                    continue
                if any(abs(y - seen) <= _ATTACK_ROW_TOLERANCE for seen in rows):
                    continue
                rows.append(y)
        if not rows and side in ("top", "bottom"):
            self._log(f"deploy side {side} had no rows; using all rows")
            return self._attack_rows(base, "all")
        return rows

    def _deploy_targets(self, base: dict, side: str = "all") -> list:
        try:
            surr_y = self._button_coords(self.config.SURR_BTN)[1]
        except Exception:
            surr_y = self.window.win_top + (self.window.win_height or 1)
        limit = surr_y - _ATTACK_ROW_MARGIN

        left = self.window.win_left
        top = self.window.win_top
        width = self.window.win_width or 1
        height = self.window.win_height or 1
        x_lo, x_hi = _SIDE_X_BOUNDS.get(side, _SIDE_X_BOUNDS["all"])
        y_lo, y_hi = _SIDE_Y_BOUNDS.get(side, _SIDE_Y_BOUNDS["all"])
        band_x_min = left + width * x_lo
        band_x_max = left + width * x_hi
        band_y_min = top + height * y_lo
        band_y_max = top + height * y_hi

        safe_x_min = left
        safe_x_max = self.window.win_width - _SWEEP_RIGHT_MARGIN

        store_centres = []
        for cls in (self.config.GOLDS, self.config.ELIXIRS, self.config.D_ELIXIRS):
            for item in base.get(cls, []):
                try:
                    store_centres.append((float(item[0]), float(item[1])))
                except (TypeError, IndexError, ValueError):
                    continue

        directions = [
            (
                math.cos(math.radians(45 * i - 90)),
                math.sin(math.radians(45 * i - 90)),
            )
            for i in range(8)
        ]
        map_cx = left + width / 2.0
        map_cy = top + height / 2.0
        keep_clear = _TARGET_RING_RADIUS - 10

        points = []
        for cx, cy in store_centres:
            if cy >= limit:
                continue
            if not (
                band_x_min <= cx <= band_x_max
                and band_y_min <= cy <= band_y_max
            ):
                continue

            vx, vy = map_cx - cx, map_cy - cy
            if vx == 0 and vy == 0:
                start = 0
            else:
                start = max(
                    range(8),
                    key=lambda i: directions[i][0] * vx
                    + directions[i][1] * vy,
                )
            order = []
            for step in range(8):
                if step == 0:
                    index = start
                elif step % 2:
                    index = start + (step + 1) // 2
                else:
                    index = start - step // 2
                order.append(index % 8)

            added = 0
            for index in order:
                if added >= _TARGETS_PER_STORE:
                    break
                dx, dy = directions[index]
                for radius in _TARGET_RINGS:
                    px = int(round(cx + dx * radius))
                    py = int(round(cy + dy * radius))
                    if not (safe_x_min <= px <= safe_x_max):
                        continue
                    if py >= limit:
                        continue
                    if any(
                        (px - ox) ** 2 + (py - oy) ** 2 < keep_clear ** 2
                        for ox, oy in store_centres
                        if (ox, oy) != (cx, cy)
                    ):
                        continue
                    if any(
                        (px - seen[0]) ** 2 + (py - seen[1]) ** 2
                        < _TARGET_SPREAD ** 2
                        for seen in points
                    ):
                        continue
                    points.append((px, py))
                    added += 1
                    break

        points.sort(key=lambda p: (p[1], p[0]))
        points = points[:_TARGETS_MAX]

        if not points and side != "all":
            self._log(f"deploy side {side} had no stores; using all sides")
            return self._deploy_targets(base, "all")
        return points

    def _sweep_rows(self, rows: list, side: str = "all") -> int:
        started = time.time()
        total_taps = 0
        x_points = list(
            range(
                self.window.win_left,
                self.window.win_width - _SWEEP_RIGHT_MARGIN,
                _SWEEP_STEP,
            )
        )
        x_lo, x_hi = _SIDE_X_BOUNDS.get(side, _SIDE_X_BOUNDS["all"])
        if (x_lo, x_hi) != (0.0, 1.0):
            span = max(0, self.window.win_width - self.window.win_left)
            start = self.window.win_left + int(round(span * x_lo))
            end = self.window.win_left + int(round(span * x_hi))
            end = min(end, self.window.win_width - _SWEEP_RIGHT_MARGIN)
            side_points = list(range(start, end, _SWEEP_STEP))
            if side_points:
                x_points = side_points
            else:
                logger.warning(
                    f"deploy side {side} produced no x points; using full width"
                )
        for k in rows:
            points = [(a, k) for a in x_points]
            logger.debug(f"tap sweep row y={k}: {len(points)} taps")
            try:
                self.auto.tap_many(points)
            except Exception as e:
                logger.error(f"deploy tap failed on row y={k}: {e}")
                raise
            total_taps += len(points)
            self._log(f"swept row y={k}: {len(points)} taps")
        self._log(
            f"attack sweep done: {len(rows)} row(s), {total_taps} taps "
            f"in {time.time() - started:.1f}s"
        )
        return total_taps

    def _deploy_slots(self, order=None) -> list:
        configured = self.troops.get("slots") or [self.config.S_GOBLIN]
        if not isinstance(configured, (list, tuple)):
            configured = [configured]
        configured = list(configured)
        if isinstance(order, (list, tuple)):
            configured = [n for n in order if n in configured] + configured
        slots = []
        for name in configured:
            if name in slots:
                continue
            if name in self.controls:
                slots.append(name)
            elif name not in self._uncalibrated_slots_logged:
                self._uncalibrated_slots_logged.add(name)
                self._log(f"troop slot {name} not calibrated; skipping")
        return slots or [self.config.S_GOBLIN]

    def _hold_rows(self, rows: list) -> int:
        try:
            hold_seconds = float(self.troops.get("hold_seconds", 0.6))
        except (TypeError, ValueError):
            hold_seconds = 0.6
        try:
            budget = int(self.troops.get("max_hold_points", 6))
        except (TypeError, ValueError):
            budget = 6
        budget = max(0, budget)
        rows = list(rows)

        if not rows or budget == 0:
            self._log(
                f"hold deploy: 0 hold(s) x {hold_seconds:g}s "
                f"over {len(rows)} row(s)"
            )
            return 0

        _, surr_y = self._button_coords(self.config.SURR_BTN)
        limit = surr_y - _ATTACK_ROW_MARGIN
        x_left = self.window.win_left + int(self.window.win_width * _HOLD_LEFT_INSET)
        x_right = self.window.win_width - _SWEEP_RIGHT_MARGIN

        per_row = [0] * len(rows)
        for i in range(budget):
            per_row[i % len(rows)] += 1

        total = 0
        for index, y in enumerate(rows):
            count = per_row[index]
            if count == 0:
                continue
            if y >= limit:
                logger.error(
                    f"refusing hold at y={y} (surrender guard: y >= {limit})"
                )
                continue
            for i in range(count):
                if count == 1:
                    x = (x_left + x_right) // 2
                else:
                    x = int(round(x_left + (x_right - x_left) * i / (count - 1)))
                logger.debug(f"hold deploy at ({x}, {y}) for {hold_seconds:g}s")
                try:
                    self.auto.hold(x, y, seconds=hold_seconds)
                except Exception as e:
                    logger.error(f"deploy hold failed at ({x}, {y}): {e}")
                    raise
                total += 1

        self._log(
            f"hold deploy: {total} hold(s) x {hold_seconds:g}s "
            f"over {len(rows)} row(s)"
        )
        return total

    def _initiate_attack(self, base: dict, plan: dict | None = None):
        plan = plan if isinstance(plan, dict) else {}
        self._last_attack_stats = {}
        if self._stop_event.is_set():
            self._log("attack skipped: stop requested before deploy")
            self._last_attack_stats = {
                "waves": 0,
                "taps": 0,
                "holds": 0,
                "elapsed_s": 0.0,
            }
            return

        drain_ratio = float(self.attack.get("drain_ratio", 0.8))
        time_limit = float(self.attack.get("time_limit", 90.0))
        if "surrender_after_s" in plan:
            try:
                brain_limit = float(plan["surrender_after_s"])
            except (TypeError, ValueError):
                brain_limit = None
            if brain_limit is not None:
                timed = max(_MIN_SURRENDER_S, min(_MAX_SURRENDER_S, brain_limit))
                if abs(timed - time_limit) > 0.5:
                    self._log(
                        f"brain surrender_after_s {timed:g}s "
                        f"(config {time_limit:g}s)"
                    )
                time_limit = timed
        side = str(plan.get("target_side") or "all").strip().lower()
        if side not in _SIDE_X_BOUNDS:
            side = "all"
        wave_settle = self.waits.get("attack_wave_settle", 5.0)

        counts = self._provider_counts(base)
        total = sum(counts.values())
        self._log(
            f"providers at scan: {total} "
            f"(gold {counts[self.config.GOLDS]} / "
            f"elixir {counts[self.config.ELIXIRS]} / "
            f"dark {counts[self.config.D_ELIXIRS]})"
        )

        current = base
        started = None
        wave = 0
        remaining = total
        total_taps = 0
        total_holds = 0
        slots = self._deploy_slots(plan.get("deploy_order"))
        deploy_mode = self.troops.get("deploy_mode", "tap")
        if side != "all":
            self._log(f"deploy side: {side}")
        if slots != self._deploy_slots():
            self._log(f"deploy order (brain): {slots}")
        while True:
            wave += 1
            rows = self._attack_rows(current, side)
            self._log(f"wave {wave}: sweeping {len(rows)} row(s)")
            points = []
            if deploy_mode == "target" and wave == 1:
                store_count = sum(
                    len(current.get(cls, []))
                    for cls in (
                        self.config.GOLDS,
                        self.config.ELIXIRS,
                        self.config.D_ELIXIRS,
                    )
                )
                points = self._deploy_targets(current, side)
                self._log(
                    f"wave {wave}: {len(points)} ground target(s) from "
                    f"{store_count} store(s)"
                )
                if len(points) < 20:
                    self._log(
                        f"wave {wave}: sparse target set ({len(points)}); "
                        "adding row sweep"
                    )
            elif deploy_mode == "target":
                self._log(
                    f"wave {wave}: flood sweep {len(rows)} row(s)"
                )
            legacy = deploy_mode == "tap" and slots == [self.config.S_GOBLIN]
            if started is None:
                started = time.time()
            for slot_index, slot in enumerate(slots):
                gx, gy = self._button_coords(slot)
                self._log(f"tap {slot} (deploy) at ({gx}, {gy})")
                self.auto.click(gx, gy)
                if not legacy:
                    self._sleep(0.3, why="troop slot settle")
                if deploy_mode == "hold":
                    if self._stop_event.is_set():
                        break
                    total_holds += self._hold_rows(rows)
                elif deploy_mode == "target" and wave == 1 and points:
                    self.auto.tap_many(points)
                    total_taps += len(points)
                    if len(points) < 20:
                        total_taps += self._sweep_rows(rows, side)
                else:
                    total_taps += self._sweep_rows(rows, side)
                if slot_index < len(slots) - 1:
                    self._sleep(0.2, why="troop slot switch")

            self._sleep(wave_settle, why="attack wave settle")

            if self._stop_event.is_set():
                self._log("stop requested between waves; surrendering")
                break

            elapsed = time.time() - started
            if elapsed >= time_limit:
                self._log(
                    f"attack time limit reached ({elapsed:.0f}s >= "
                    f"{time_limit:g}s, wave {wave}); surrendering"
                )
                break

            if total == 0:
                self._log("no providers detected; not re-scanning")
                break

            try:
                current = self._scan_base_layout()
            except ScanFailed as e:
                self._log(f"post-wave rescan failed ({e}); surrendering")
                break

            counts = self._provider_counts(current)
            remaining = sum(counts.values())
            drained = 1.0 - (remaining / total)
            self._log(
                f"wave {wave}: providers {remaining}/{total} remaining "
                f"({drained * 100:.0f}% drained)"
            )
            if drained >= drain_ratio:
                self._log(
                    f"drain target reached ({drained * 100:.0f}% >= "
                    f"{drain_ratio * 100:.0f}%) after {wave} wave(s)"
                )
                break

            if side != "all":
                self._log(
                    f"drain {drained * 100:.0f}% < {drain_ratio * 100:.0f}% on "
                    f"side {side}; widening next wave to all sides"
                )
                side = "all"

        drained = None
        if total > 0:
            now = self._count_providers_now()
            fresh = now is not None
            if fresh:
                remaining = now
            remaining = max(0, min(remaining, total))
            drained = 1.0 - (remaining / total)
            self._log(
                f"drain at surrender: providers {remaining}/{total} remaining "
                f"({drained * 100:.0f}% drained"
                + ("" if fresh else ", last known")
                + ")"
            )

        settle = self.waits.get("attack_settle", 5.0)
        self._sleep(settle, why="attack settle before surrender")
        elapsed_s = time.time() - started if started is not None else 0.0
        self._last_attack_stats = {
            "waves": wave,
            "taps": total_taps,
            "holds": total_holds,
            "elapsed_s": round(elapsed_s, 3),
        }
        return drained

    def _farm_loop(self, targets, goals):
        gtarget, etarget, detarget = targets
        g_goal, e_goal, de_goal = goals

        st_time = time.time()
        totals = {"gold": 0, "elixir": 0, "delixir": 0}
        raid_drains = []

        self._set_status("loading")
        try:
            mode = "AI with rule fallback" if self.ai_enabled else "rule-based"
            self._log(f"strategy: {mode}")
            self._log("Loading YOLO + OCR models...")
            self.vision.load()
            self._log(f"vision device: {self.vision.device_name}")
            frame = self.window.capture_window()
            self._log("Warming up YOLO + OCR...")
            self.vision.warmup(frame)
            self._log(
                f"models ready; stop_requested={self._stop_event.is_set()}; "
                f"frame={self.window.win_width}x{self.window.win_height}; "
                f"initial capture={self.window.last_capture_ms:.0f}ms"
            )
            self._publish_dashboard(frame=frame, stage="Models ready")

            self._set_status("farming")
            scan_streak = 0
            scan_limit = max(1, int(self.recovery.get("scan_restart_limit", 3)))
            zero_loot_streak = 0

            while not self._stop_event.is_set():
                if (
                    totals["gold"] > g_goal
                    and totals["elixir"] > e_goal
                    and totals["delixir"] > de_goal
                ):
                    e_time = time.time()
                    mins = (e_time - st_time) // 60
                    msg = f"Completed in {mins} min"
                    self._log(msg)
                    self._alert(msg)
                    self._notify(
                        "COC Auto-Farm: completed",
                        f"Gold {totals['gold']} | Elixir {totals['elixir']} | "
                        f"Dark {totals['delixir']}, in {mins} min"
                        + self._drain_average(raid_drains),
                        success=True,
                    )
                    break

                for btn in [
                    self.config.ATTACK_BTN_HOME,
                    self.config.FIND_BTN,
                    self.config.ATTACK_BTN,
                ]:
                    if self._stop_event.is_set():
                        break
                    self._press_button(btn)

                if self._stop_event.is_set():
                    break

                gave_up = False
                search = True
                while search and not self._stop_event.is_set():
                    try:
                        base = self._scan_base_layout()
                        base_treasure = self._scan_treasures()
                    except ScanFailed as e:
                        scan_streak += 1
                        self._notify(
                            f"COC Auto-Farm: scan failed "
                            f"({scan_streak}/{scan_limit})",
                            f"{e}"
                            + (" (stopping)" if scan_streak >= scan_limit else "")
                            + (
                                self._drain_average(raid_drains)
                                if scan_streak >= scan_limit
                                else ""
                            ),
                        )
                        if isinstance(e, TreasureScanFailed) and scan_streak == 1:
                            self._log(
                                f"resource scan failed ({scan_streak}/{scan_limit}): "
                                f"{e}; tapping surr and restarting phase 1"
                            )
                            self._press_button(self.config.SURR_BTN)
                        else:
                            self._log(
                                f"scan failed ({scan_streak}/{scan_limit}): {e}; "
                                "backing out (retry_btn_1/2/3) and restarting"
                            )
                            self._press_button(self.config.RETRY_BTN_1)
                            self._press_button(self.config.RETRY_BTN_2)
                            self._press_button(self.config.RETRY_BTN_3)
                        if scan_streak >= scan_limit:
                            mins = (time.time() - st_time) // 60
                            self._alert(
                                f"error: {e}, in {mins} min "
                                f"(scan failed {scan_streak}x, stopping)"
                            )
                            gave_up = True
                        search = False
                        break

                    scan_streak = 0

                    current_gold = int(base_treasure.get(self.config.GOLD, 0))
                    current_elixir = int(base_treasure.get(self.config.ELIXIR, 0))
                    current_delixir = int(base_treasure.get(self.config.D_ELIXIR, 0))

                    self._log(
                        f"base loot: Gold {current_gold} | "
                        f"Elixir {current_elixir} | Dark {current_delixir}"
                    )

                    alt_plan = {}
                    state = {}
                    try:
                        state = self._build_raid_state(
                            base,
                            base_treasure,
                            targets=(gtarget, etarget, detarget),
                        )
                        plan = self.brain.plan(state)
                        alt_plan = {}
                        if self.ai_enabled and self.brain_alt is not None:
                            alt_plan = self.brain_alt.plan(state)
                            self._log(
                                f"brain alt ({self.brain_alt.describe()}): {alt_plan}"
                            )
                            if not alt_plan:
                                reason = getattr(
                                    self.brain_alt, "last_error", ""
                                ) or "no plan"
                                self._log(
                                    f"brain alt unavailable ({reason}); "
                                    "rule plan for this raid"
                                )
                        recorder = self.replay.start(state, plan)
                        recorder.event(
                            "scan",
                            gold=current_gold,
                            elixir=current_elixir,
                            delixir=current_delixir,
                        )
                        if (
                            recorder is not None
                            and self.ai_enabled
                            and self.brain_alt is not None
                        ):
                            recorder.event(
                                "plan_alt",
                                name=self.brain_alt.name,
                                plan=alt_plan,
                                error=str(
                                    getattr(self.brain_alt, "last_error", "") or ""
                                ),
                            )
                        self._log(f"brain plan (rule): {plan}")
                    except Exception as e:
                        logger.warning(f"shadow brain/replay setup failed: {e}")
                        recorder = None

                    active_plan = {}
                    if alt_plan and self.brain_alt is not None:
                        active_plan = self._validate_plan(alt_plan, state)
                        self._log(
                            f"brain plan active ({self.brain_alt.name}): {active_plan}"
                        )
                        if recorder is not None:
                            recorder.event(
                                "plan_active",
                                plan=active_plan,
                                side_requested=self._side_requested,
                            )

                    brain_skip = bool(active_plan.get("skip_base"))
                    if brain_skip and self._brain_skip_streak >= _MAX_BRAIN_SKIP_STREAK:
                        self._log(
                            f"brain skipped {self._brain_skip_streak} bases in a "
                            "row; attacking this one anyway"
                        )
                        brain_skip = False

                    if (
                        current_gold >= gtarget
                        and current_delixir >= detarget
                        and current_elixir >= etarget
                        and not brain_skip
                    ):
                        self._brain_skip_streak = 0
                        self._publish_dashboard(screen="battle", stage="Executing raid")
                        if recorder is not None:
                            recorder.event("attack_start")
                        drained = self._initiate_attack(base, active_plan)

                        self._press_button(self.config.SURR_BTN)
                        self._press_button(self.config.SURR_CONF)

                        post_game = self._scan_post_game(
                            base_loot={
                                "gold": current_gold,
                                "elixir": current_elixir,
                                "delixir": current_delixir,
                            }
                        )
                        if not post_game:
                            logger.warning(
                                "post-battle summary produced no loot numbers; "
                                "this raid is counted as a zero gain"
                            )
                            self._log("loot read failed for this raid")
                        gains = [
                            (
                                "gold",
                                int(post_game.get(self.config.POST_GOLD, 0)),
                                current_gold,
                            ),
                            (
                                "elixir",
                                int(post_game.get(self.config.POST_ELIXIR, 0)),
                                current_elixir,
                            ),
                            (
                                "delixir",
                                int(post_game.get(self.config.POST_D_ELIXIR, 0)),
                                current_delixir,
                            ),
                        ]
                        clamped = {}
                        for label, raw, base_value in gains:
                            value = raw
                            if base_value > 0:
                                cap = int(base_value * _LOOT_CLAMP_SLACK)
                                if raw > cap:
                                    value = cap
                                    msg = (
                                        f"loot read clamped: {label} "
                                        f"{raw} -> {value}"
                                    )
                                    logger.info(msg)
                                    self._log(msg)
                            clamped[label] = value
                        g_gain = clamped["gold"]
                        e_gain = clamped["elixir"]
                        de_gain = clamped["delixir"]
                        totals["gold"] += g_gain
                        totals["elixir"] += e_gain
                        totals["delixir"] += de_gain
                        self._log(
                            f"loot this raid: Gold +{g_gain} | Elixir +{e_gain} | "
                            f"Dark +{de_gain}  (totals Gold {totals['gold']} | "
                            f"Elixir {totals['elixir']} | Dark {totals['delixir']})"
                        )
                        self._show_totals(totals)

                        if recorder is not None:
                            recorder.finish(
                                status="attacked",
                                loot_gain={
                                    "gold": g_gain,
                                    "elixir": e_gain,
                                    "delixir": de_gain,
                                },
                                drained=drained,
                                attack=self._last_attack_stats,
                            )

                        if drained is not None:
                            raid_drains.append(drained)
                        self._publish_dashboard(
                            screen="post_battle",
                            stage="Raid completed",
                            loot=post_game,
                            raid={
                                "timestamp": time.time(),
                                "status": "attacked",
                                "base_loot": {
                                    "gold": current_gold,
                                    "elixir": current_elixir,
                                    "delixir": current_delixir,
                                },
                                "loot_gain": {
                                    "gold": g_gain,
                                    "elixir": e_gain,
                                    "delixir": de_gain,
                                },
                                "drained": drained,
                                "attack": dict(self._last_attack_stats),
                            },
                        )

                        self._raid_history.append(
                            {
                                "gold": g_gain,
                                "elixir": e_gain,
                                "delixir": de_gain,
                                "drained": drained,
                                "base_loot": {
                                    "gold": current_gold,
                                    "elixir": current_elixir,
                                    "delixir": current_delixir,
                                },
                            }
                        )
                        del self._raid_history[:-_HISTORY_MAX]

                        if g_gain == 0 and e_gain == 0 and de_gain == 0:
                            zero_loot_streak += 1
                        else:
                            zero_loot_streak = 0

                        self._press_button(self.config.RETURN_HOME)

                        if zero_loot_streak >= scan_limit:
                            mins = (time.time() - st_time) // 60
                            self._alert(
                                f"error: no loot read for {zero_loot_streak} raids "
                                f"in a row, in {mins} min"
                            )
                            self._notify(
                                "COC Auto-Farm: no loot read",
                                f"{zero_loot_streak} raids in a row collected 0 "
                                f"loot; stopping after {mins} min"
                                + self._drain_average(raid_drains),
                            )
                            gave_up = True

                        search = False
                    else:
                        if brain_skip:
                            self._brain_skip_streak += 1
                            if recorder is not None:
                                recorder.event(
                                    "brain_skip",
                                    streak=self._brain_skip_streak,
                                )
                        if recorder is not None:
                            recorder.finish(status="skipped")
                        self._publish_dashboard(
                            screen="scouting",
                            stage="Base skipped",
                            loot={
                                "gold": current_gold,
                                "elixir": current_elixir,
                                "delixir": current_delixir,
                            },
                            raid={
                                "timestamp": time.time(),
                                "status": "skipped",
                                "base_loot": {
                                    "gold": current_gold,
                                    "elixir": current_elixir,
                                    "delixir": current_delixir,
                                },
                                "loot_gain": {"gold": 0, "elixir": 0, "delixir": 0},
                                "drained": None,
                                "attack": {},
                            },
                        )
                        self._press_button(self.config.NXT_BTN)

                if gave_up:
                    self._set_status("error")
                    break

        except RuntimeError as e:
            if str(e) == "Bot stopped":
                self._log(
                    "Bot stopped cleanly"
                    + (
                        f" (stop source: {self._stop_request_source})"
                        if self._stop_request_source
                        else " (no recorded stop() caller; inspect local code/version)"
                    )
                )
            else:
                self._log(f"error: {e}")
                mins = (time.time() - st_time) // 60
                self._alert(f"error: {e}, in {mins} min")
                self._notify(
                    "COC Auto-Farm: error",
                    f"{e} (after {mins} min)" + self._drain_average(raid_drains),
                )
            self._set_status("error")
        except Exception as e:
            self._log(f"error: {e}")
            mins = (time.time() - st_time) // 60
            self._alert(f"error: {e}, in {mins} min")
            self._notify(
                "COC Auto-Farm: error",
                f"{e} (after {mins} min)" + self._drain_average(raid_drains),
            )
            self._set_status("error")
        finally:
            self._set_status("stopped")
            self._show_totals(totals)
            self._log(self._drain_summary(raid_drains))
