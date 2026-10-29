import logging
import time

logger = logging.getLogger(__name__)

SCREENS = ("home", "find_match", "scouting", "battle", "post_battle", "unknown")

SCREEN_PRIORITY = [
    (
        "post_battle",
        frozenset({"return_home", "post_gold", "post_elixir", "post_d_elixir"}),
    ),
    ("battle", frozenset({"surr_conf", "end_btn"})),
    ("scouting", frozenset({"nxt_btn"})),
    ("find_match", frozenset({"find_btn", "attack_btn"})),
    ("home", frozenset({"attack_btn_home"})),
]

SCREEN_CLASSES = {
    screen_name: classes for screen_name, classes in SCREEN_PRIORITY
}
SCREEN_CLASSES["unknown"] = frozenset()

CANDIDATE_CLASSES = tuple(
    sorted({class_name for _, classes in SCREEN_PRIORITY for class_name in classes})
)


class ScreenDetector:

    def __init__(self, vision, window, conf=None):
        self.vision = vision
        self.window = window
        self.conf = conf

    @staticmethod
    def _screen_from_details(details):
        present = {
            class_name for class_name, hits in details.items() if hits
        }
        for screen_name, classes in SCREEN_PRIORITY:
            if present & classes:
                return screen_name
        return "unknown"

    def detect(self, frame=None, details=None):
        try:
            if details is not None:
                return self._screen_from_details(details)

            if frame is None:
                frame = self.window.capture_window()

            screen = self._screen_from_details(
                self.vision.detect(frame, CANDIDATE_CLASSES, conf=self.conf)
            )
            if screen != "unknown":
                return screen

            configured_conf = self.conf
            if configured_conf is None:
                configured_conf = getattr(
                    getattr(self.vision, "config", None), "conf", 0.5
                )
            try:
                configured_conf = float(configured_conf)
            except (TypeError, ValueError):
                configured_conf = 0.5

            thresholds = []
            for candidate in (
                min(configured_conf * 0.5, 0.25),
                0.15,
                0.08,
                0.04,
            ):
                candidate = max(0.01, float(candidate))
                if candidate < configured_conf and candidate not in thresholds:
                    thresholds.append(candidate)

            attempted = []
            for fallback_conf in thresholds:
                attempted.append(fallback_conf)
                fallback = self.vision.detect(
                    frame, CANDIDATE_CLASSES, conf=fallback_conf
                )
                screen = self._screen_from_details(fallback)
                if screen != "unknown":
                    logger.info(
                        "screen recognized as %s at fallback confidence %.2f",
                        screen, fallback_conf
                    )
                    return screen

            logger.debug(
                "screen remains unknown; no UI marker detected at confidence "
                "thresholds %s",
                [round(configured_conf, 3)] + [round(v, 3) for v in attempted]
            )
            return "unknown"
        except Exception:
            logger.debug("screen detection failed", exc_info=True)
            return "unknown"

    def wait_for(self, names, timeout_s, poll_s=0.3, stop_event=None):
        if isinstance(names, str):
            names = (names,)
        wanted = frozenset(names)

        timeout_s = max(0.0, float(timeout_s))
        deadline = time.monotonic() + timeout_s
        interval = float(poll_s)
        if timeout_s > 0 and interval > timeout_s / 2.0:
            interval = timeout_s / 2.0

        while True:
            if stop_event is not None and stop_event.is_set():
                return None

            screen = self.detect()
            if screen in wanted:
                return screen

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None

            delay = min(interval, remaining)
            if delay <= 0:
                continue
            if stop_event is not None:
                if stop_event.wait(delay):
                    return None
            else:
                time.sleep(delay)

    def describe(self) -> str:
        return (
            f"screen detector conf={self.conf} "
            f"probe={len(CANDIDATE_CLASSES)} classes"
        )
