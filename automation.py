import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

SCREEN_SIG_SIZE = 32
VERIFY_REGION = 240


class AutomationController:

    def __init__(self, adb):
        if adb is None:
            raise ValueError("AutomationController requires a connected EmulatorAdb (ADB-only mode)")
        self.adb = adb

    def click(self, x, y=None):
        if y is None:
            x, y = x
        if not self.adb.tap(int(x), int(y)):
            raise RuntimeError("ADB tap failed")

    def tap_many(self, points):
        points = list(points)
        if not points:
            return
        if not self.adb.tap_many(points):
            raise RuntimeError("ADB batch tap failed")

    def hold(self, x, y=None, seconds=0.6):
        if y is None:
            x, y = x
        if not self.adb.hold(int(x), int(y), int(round(seconds * 1000))):
            raise RuntimeError("ADB hold failed")

    def send_key(self, key):
        keycode = _KEYEVENT_MAP.get(key)
        if keycode is None:
            raise ValueError(f"Unsupported key for ADB: {key!r}")
        if not self.adb.keyevent(keycode):
            raise RuntimeError(f"ADB keyevent failed for {key!r}")

    def signature(self, x, y, region=VERIFY_REGION) -> bytes | None:
        try:
            if region is None:
                frame = self.adb.screencap()
                if frame is None or frame.size == 0:
                    return None
                crop = frame
            else:
                side = int(region)
                crop = self.adb.screencap_region(
                    int(x) - side // 2, int(y) - side // 2, side, side
                )
                if crop is None or crop.size == 0:
                    return None
            if crop.size == 0:
                return None
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(
                gray,
                (SCREEN_SIG_SIZE, SCREEN_SIG_SIZE),
                interpolation=cv2.INTER_AREA,
            )
        except Exception:
            return None
        return small.tobytes()

    @staticmethod
    def changed(before, after, threshold=2.0) -> bool:
        if before is None or after is None:
            return False
        try:
            a = np.frombuffer(before, dtype=np.uint8)
            b = np.frombuffer(after, dtype=np.uint8)
            if a.size != b.size or a.size == 0:
                return False
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
            return float(np.mean(diff)) > threshold
        except Exception:
            return False

    def click_verified(
        self,
        x,
        y,
        timeout_s=1.2,
        poll_s=0.15,
        region=VERIFY_REGION,
        retries=0,
        threshold=2.0,
        verify_global=False,
    ) -> bool:
        x, y = int(x), int(y)
        verification_region = None if verify_global else region
        try:
            before = self.signature(x, y, verification_region)
        except Exception:
            before = None
        if before is None:
            logger.error(
                "refusing tap (%d, %d): pre-click screenshot is unavailable", x, y
            )
            return False

        if not self.adb.tap(x, y):
            logger.error("tap (%d, %d) failed at the ADB layer", x, y)
            return False
        logger.debug("tap (%d, %d) attempt 0", x, y)
        if self._poll_change(
            x, y, before, timeout_s, poll_s, verification_region, threshold
        ):
            logger.debug("tap (%d, %d) verified", x, y)
            return True

        offsets = ((6, -6), (-6, 6))
        for attempt in range(1, int(retries) + 1):
            dx, dy = offsets[(attempt - 1) % len(offsets)]
            rx, ry = x + dx, y + dy
            logger.info(
                f"tap ({x}, {y}) no change after {timeout_s:g}s; retry {attempt}"
            )
            if not self.adb.tap(rx, ry):
                logger.error("retry tap (%d, %d) failed at the ADB layer", rx, ry)
                return False
            logger.debug("tap (%d, %d) attempt %d", rx, ry, attempt)
            if self._poll_change(
                rx, ry, before, timeout_s, poll_s, verification_region, threshold
            ):
                logger.debug("retry tap (%d, %d) verified", rx, ry)
                return True

        return False

    def _poll_change(self, x, y, before, timeout_s, poll_s, region, threshold):
        deadline = time.time() + float(timeout_s)
        while True:
            try:
                after = self.signature(x, y, region)
                if self.changed(before, after, threshold):
                    return True
            except Exception:
                pass
            if time.time() >= deadline:
                return False
            time.sleep(poll_s)


_KEYEVENT_MAP = {
    "back": 4,
    "home": 3,
    "enter": 66,
    "space": 62,
    "escape": 111,
}