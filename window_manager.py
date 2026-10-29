import logging
import shutil
import time

from adb import VideoCaptureStream

logger = logging.getLogger(__name__)


class WindowManager:

    def __init__(self, adb):
        if adb is None:
            raise ValueError("WindowManager requires a connected EmulatorAdb (ADB-only mode)")
        self.adb = adb
        self.win_left = 0
        self.win_top = 0
        self.win_width = 0
        self.win_height = 0
        self.last_capture_ms = 0.0
        self.video_stream = None
        self._video_fallback_logged = False

    def focus_window(self):
        pass

    def start_video_capture(self):
        if self.video_stream is not None:
            return True

        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            logger.warning(
                "FFmpeg was not found on PATH; using slower ADB screenshot capture"
            )
            return False

        started = time.perf_counter()
        frame = self.adb.screencap()
        self.last_capture_ms = (time.perf_counter() - started) * 1000.0
        if frame is None:
            logger.warning(
                "Cannot start video capture because the initial screenshot failed: %s",
                self.adb.last_error,
            )
            return False

        self.win_height, self.win_width = frame.shape[:2]
        self.win_left = 0
        self.win_top = 0
        stream = VideoCaptureStream(
            self.adb.adb,
            self.adb.serial,
            self.win_width,
            self.win_height,
            ffmpeg_path,
            fps=20,
        )
        if not stream.start():
            logger.warning(
                "Live video capture unavailable (%s); using ADB screenshot capture",
                stream.last_error,
            )
            return False

        self.video_stream = stream
        logger.info(
            "Live H.264 video capture started (%dx%d at %d fps)",
            self.win_width,
            self.win_height,
            stream.fps,
        )
        return True

    @property
    def video_stream_active(self):
        return self.video_stream is not None and self.video_stream.is_active

    def stop_video_capture(self):
        if self.video_stream is not None:
            self.video_stream.stop()
            self.video_stream = None

    def get_rect(self):
        return self.capture_window()

    def capture_window(self):
        if self.video_stream is not None:
            frame, age_ms = self.video_stream.capture()
            if frame is not None:
                self.last_capture_ms = age_ms
                self.win_height, self.win_width = frame.shape[:2]
                self.win_left = 0
                self.win_top = 0
                return frame
            if not self._video_fallback_logged:
                logger.warning(
                    "Live video frame unavailable (%s); falling back to ADB screenshots",
                    self.video_stream.last_error or "stream frame is stale",
                )
                self._video_fallback_logged = True

        started = time.perf_counter()
        frame = self.adb.screencap()
        self.last_capture_ms = (time.perf_counter() - started) * 1000.0
        if frame is None:
            raise RuntimeError(f"ADB screencap failed: {self.adb.last_error}")
        self.win_height, self.win_width = frame.shape[:2]
        self.win_left = 0
        self.win_top = 0
        return frame