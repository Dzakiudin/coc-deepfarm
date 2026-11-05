_KEYEVENT_MAP = {
    "back": 4,
    "home": 3,
    "enter": 66,
    "space": 62,
    "escape": 111,
}


class AutomationController:

    def __init__(self, adb):
        if adb is None:
            raise ValueError(
                "AutomationController requires a connected EmulatorAdb (ADB-only mode)"
            )
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
