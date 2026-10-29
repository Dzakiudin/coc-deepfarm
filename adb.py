import glob
import logging
import os
import shutil
import socket
import subprocess
import threading
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoCaptureStream:

    def __init__(self, adb_path, serial, width, height, ffmpeg_path, fps=20):
        self.adb_path = adb_path
        self.serial = serial
        self.width = int(width)
        self.height = int(height)
        self.ffmpeg_path = ffmpeg_path
        self.fps = int(fps)
        self._stop_event = threading.Event()
        self._first_frame = threading.Event()
        self._frame_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._frame = None
        self._frame_at = 0.0
        self._adb_process = None
        self._decoder_process = None
        self._thread = None
        self.last_error = ""

    @property
    def is_active(self):
        if self._thread is None or not self._thread.is_alive():
            return False
        with self._frame_lock:
            frame_at = self._frame_at
        return frame_at > 0 and time.monotonic() - frame_at <= 0.75

    def start(self, timeout=6.0):
        if self._thread is not None and self._thread.is_alive():
            return self._first_frame.wait(timeout)

        self._stop_event.clear()
        self._first_frame.clear()
        self.last_error = ""
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="AndroidVideoCapture",
            daemon=True,
        )
        self._thread.start()
        if self._first_frame.wait(timeout):
            return True

        self.last_error = self.last_error or "timed out waiting for the first video frame"
        self.stop()
        return False

    def capture(self, max_age=0.75):
        with self._frame_lock:
            frame = self._frame
            frame_at = self._frame_at
        if frame is None:
            return None, None

        age_ms = (time.monotonic() - frame_at) * 1000.0
        if age_ms > max_age * 1000.0:
            with self._process_lock:
                adb_process = self._adb_process
            self._terminate(adb_process)
            return None, age_ms
        return frame.copy(), age_ms

    def stop(self):
        self._stop_event.set()
        with self._process_lock:
            adb_process = self._adb_process
            decoder_process = self._decoder_process
        self._terminate(adb_process)
        self._terminate(decoder_process)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=3.0)

    @staticmethod
    def _terminate(process):
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        except OSError:
            return

    def _read_frame(self, stream):
        remaining = self.width * self.height * 3
        frame = bytearray(remaining)
        offset = 0
        while offset < remaining and not self._stop_event.is_set():
            chunk = stream.read(min(262144, remaining - offset))
            if not chunk:
                return None
            frame[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        return frame if offset == remaining else None

    def _capture_loop(self):
        while not self._stop_event.is_set():
            adb_process = None
            decoder_process = None
            try:
                adb_process = subprocess.Popen(
                    [
                        self.adb_path,
                        "-s",
                        self.serial,
                        "exec-out",
                        "screenrecord",
                        "--output-format=h264",
                        f"--size={self.width}x{self.height}",
                        "--bit-rate=4000000",
                        "--time-limit=0",
                        "-",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                decoder_process = subprocess.Popen(
                    [
                        self.ffmpeg_path,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "h264",
                        "-i",
                        "pipe:0",
                        "-vf",
                        f"fps={self.fps}",
                        "-an",
                        "-sn",
                        "-dn",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "bgr24",
                        "pipe:1",
                    ],
                    stdin=adb_process.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if adb_process.stdout is not None:
                    adb_process.stdout.close()
                with self._process_lock:
                    self._adb_process = adb_process
                    self._decoder_process = decoder_process

                while not self._stop_event.is_set():
                    raw_frame = self._read_frame(decoder_process.stdout)
                    if raw_frame is None:
                        if not self._stop_event.is_set():
                            raise RuntimeError("H.264 video stream ended")
                        break
                    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        self.height, self.width, 3
                    )
                    with self._frame_lock:
                        self._frame = frame
                        self._frame_at = time.monotonic()
                    self.last_error = ""
                    self._first_frame.set()
            except Exception as e:
                self.last_error = str(e)
                if not self._stop_event.is_set():
                    logger.warning("Android video capture failed: %s", e)
            finally:
                self._terminate(adb_process)
                self._terminate(decoder_process)
                with self._process_lock:
                    if self._adb_process is adb_process:
                        self._adb_process = None
                    if self._decoder_process is decoder_process:
                        self._decoder_process = None

            if not self._stop_event.is_set():
                self._stop_event.wait(1.0)


def _platform_tools_adb():
    for var in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        base = os.environ.get(var)
        if base:
            candidate = os.path.join(base, "platform-tools", "adb.exe")
            if os.path.isfile(candidate):
                return candidate
    return None


def _find_ldplayer_adb():
    search_roots = []
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    for root in search_roots:
        hits = glob.glob(os.path.join(root, "LDPlayer*", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "LDPlayer9", "adb.exe"))
        if hits:
            return hits[0]
    return None


def _find_mumu_adb():
    search_roots = []
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(var)
        if base:
            search_roots.append(base)
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    for root in search_roots:
        hits = glob.glob(
            os.path.join(root, "Netease", "MuMuPlayer*", "nx_main", "adb.exe")
        )
        if not hits:
            hits = glob.glob(
                os.path.join(root, "Netease", "MuMuPlayer*", "*", "*", "adb.exe")
            )
        if hits:
            return hits[0]
    return None


def _find_memu_adb():
    search_roots = []
    for var in ("ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(var)
        if base:
            search_roots.append(base)
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    for root in search_roots:
        hits = glob.glob(os.path.join(root, "Microvirt", "MEmu", "adb.exe"))
        if not hits:
            hits = glob.glob(os.path.join(root, "MEmu", "adb.exe"))
        if hits:
            return hits[0]
    return None


def _find_msi_adb():
    search_roots = []
    for drive in "CDEF":
        drive += ":"
        if os.path.exists(drive):
            search_roots.append(drive + os.sep)

    for root in search_roots:
        hits = []
        hits += glob.glob(os.path.join(root, "MSI App Player", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "MSI App Player", "adb.exe"))
        hits += glob.glob(
            os.path.join(root, "MSI", "MSI App Player", "*", "adb.exe")
        )
        hits += glob.glob(
            os.path.join(root, "Program Files", "MSI", "MSI App Player", "*", "adb.exe")
        )
        hits += glob.glob(
            os.path.join(
                root, "Program Files (x86)", "MSI", "MSI App Player", "*", "adb.exe"
            )
        )
        hits += glob.glob(os.path.join(root, "BlueStacks_msi5", "*", "adb.exe"))
        hits += glob.glob(os.path.join(root, "BlueStacks_msi5", "*", "HD-Adb.exe"))
        hits += glob.glob(
            os.path.join(
                root, "Users", "*", "AppData", "Local", "MSI App Player", "*", "adb.exe"
            )
        )
        hits += glob.glob(
            os.path.join(
                root, "Users", "*", "AppData", "Local", "HD-Player_msi5", "*", "adb.exe"
            )
        )
        hits += glob.glob(
            os.path.join(
                root,
                "Users",
                "*",
                "AppData",
                "Local",
                "HD-Player_msi5",
                "*",
                "HD-Adb.exe",
            )
        )
        if hits:
            return hits[0]
    return None


def find_adb():
    path = shutil.which("adb")
    if path:
        return path
    path = shutil.which("adb.exe")
    if path:
        return path
    path = _platform_tools_adb()
    if path:
        return path
    for finder in (
        _find_msi_adb,
        _find_ldplayer_adb,
        _find_mumu_adb,
        _find_memu_adb,
    ):
        path = finder()
        if path:
            return path
    return None


class EmulatorAdb:

    def __init__(self, host="127.0.0.1", port=5555, adb_path="", auto_scan=True):
        self.host = host
        self.port = int(port)
        self.adb = adb_path or find_adb()
        self.serial = f"{host}:{self.port}"
        self.auto_scan = auto_scan
        self.last_error = ""
        self._connected = False
        self._res = (0, 0)

    @property
    def connected(self):
        return self._connected

    def connect(self, timeout=15.0):
        if self.adb is None:
            self.last_error = (
                "adb not found (install Android platform-tools, or use the adb "
                "bundled with LDPlayer / MuMu Player / MEmu)"
            )
            return False

        candidates = [self.port]
        if self.auto_scan:
            extra = list(range(5555, 5600, 2))
            extra += [16384 + 32 * n for n in range(5)]
            extra += [21503 + 32 * n for n in range(3)]
            seen = {self.port}
            for p in extra:
                if p not in seen:
                    seen.add(p)
                    candidates.append(p)

        for index, port in enumerate(candidates):
            if index > 0 and not self._port_open(self.host, port):
                continue
            serial = f"{self.host}:{port}"
            self._run(["connect", serial], timeout=8)
            deadline = time.time() + (timeout if index == 0 else 1.0)
            while time.time() < deadline:
                if self._is_device(serial):
                    self.serial = serial
                    self.port = port
                    self._connected = True
                    self._refresh_resolution()
                    return True
                time.sleep(0.2)

        self.last_error = (
            f"could not reach any adb device on {self.host} "
            "(tried LDPlayer 5555-5599, MuMu 16384-16512, MEmu 21503-21567)"
        )
        return False

    def _is_device(self, serial):
        p = self._run(["devices"], timeout=8)
        if p is None or p.returncode != 0:
            return False
        for line in p.stdout.decode("utf-8", "ignore").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == serial and parts[1].strip() == "device":
                return True
        return False

    @staticmethod
    def _port_open(host, port, timeout=0.25):
        try:
            with socket.create_connection((host, int(port)), timeout=timeout):
                return True
        except OSError:
            return False

    def _run(self, args, timeout=30.0):
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            return subprocess.run(
                [self.adb, "-s", self.serial] + args,
                capture_output=True,
                timeout=timeout,
                creationflags=flags,
            )
        except subprocess.TimeoutExpired:
            self.last_error = f"adb command timed out: {' '.join(args)}"
            return None

    def _refresh_resolution(self):
        p = self._run(["shell", "wm", "size"], timeout=10)
        if p is None or p.returncode != 0:
            return
        for word in p.stdout.decode("utf-8", "ignore").replace(":", " ").split():
            if "x" in word and word.replace("x", "").isdigit():
                try:
                    w, h = (int(v) for v in word.split("x"))
                    self._res = (w, h)
                except ValueError:
                    pass
                return

    def resolution(self):
        if self._res == (0, 0):
            self._refresh_resolution()
        return self._res

    def screencap(self):
        if not self._connected:
            return None
        p = self._run(["exec-out", "screencap", "-p"], timeout=30)
        if p is None or p.returncode != 0 or not p.stdout:
            self.last_error = "screencap failed"
            return None
        frame = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.last_error = "screencap decode failed"
            return None
        return frame

    def screencap_region(self, x, y, w, h):
        frame = self.screencap()
        if frame is None:
            return None
        fh, fw = frame.shape[:2]
        x0 = max(0, min(int(x), fw))
        y0 = max(0, min(int(y), fh))
        x1 = max(0, min(int(x) + int(w), fw))
        y1 = max(0, min(int(y) + int(h), fh))
        if x1 <= x0 or y1 <= y0:
            return None
        return np.ascontiguousarray(frame[y0:y1, x0:x1]).copy()

    def tap(self, x, y):
        if not self._connected:
            return False
        p = self._run(
            ["shell", "input", "tap", str(int(x)), str(int(y))], timeout=10
        )
        return p is not None and p.returncode == 0

    def swipe(self, x1, y1, x2, y2, duration_ms=300):
        if not self._connected:
            return False
        duration_ms = int(duration_ms)
        p = self._run(
            [
                "shell",
                "input",
                "swipe",
                str(int(x1)),
                str(int(y1)),
                str(int(x2)),
                str(int(y2)),
                str(duration_ms),
            ],
            timeout=max(10.0, duration_ms / 1000.0 + 5.0),
        )
        return p is not None and p.returncode == 0

    def tap_many(self, points, timeout=None):
        if not self._connected:
            return False
        points = [(int(x), int(y)) for x, y in points]
        if not points:
            return True
        cmd = "; ".join(f"input tap {x} {y}" for x, y in points)
        p = self._run(
            ["shell", cmd],
            timeout=timeout or max(10.0, len(points) * 0.25 + 5.0),
        )
        return p is not None and p.returncode == 0

    def hold(self, x, y, duration_ms=600):
        if not self._connected:
            return False
        return self.swipe(x, y, x, y, duration_ms)

    def keyevent(self, keycode):
        if not self._connected:
            return False
        p = self._run(["shell", "input", "keyevent", str(int(keycode))], timeout=10)
        return p is not None and p.returncode == 0