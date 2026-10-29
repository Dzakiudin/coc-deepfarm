import logging
import threading
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "https://ntfy.sh"
DEFAULT_MAX_BYTES = 1_800_000


def _header(value) -> str:
    return str(value).encode("ascii", "replace").decode("ascii")


class Notifier:

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.enabled = bool(config.get("enabled"))
        self.topic = str(config.get("ntfy_topic") or "").strip()
        self.server = str(config.get("ntfy_server") or DEFAULT_SERVER).rstrip("/")
        self.token = str(config.get("ntfy_token") or "").strip()
        try:
            self.max_bytes = int(config.get("attachment_max_bytes") or DEFAULT_MAX_BYTES)
        except (TypeError, ValueError):
            self.max_bytes = DEFAULT_MAX_BYTES

    @property
    def is_configured(self) -> bool:
        return self.enabled and bool(self.topic)

    def notify(
        self,
        title: str,
        message: str,
        image_path: str | None = None,
        *,
        blocking: bool = False,
        priority: str | None = None,
        tags: str | None = None,
    ) -> bool:
        if not self.is_configured:
            return False

        args = (title, message, image_path, priority, tags)
        if blocking:
            self._send(*args)
            return True
        try:
            threading.Thread(target=self._send, args=args, daemon=True).start()
        except Exception as e:  # pragma: no cover
            logger.warning(f"could not start notifier thread: {e}")
            return False
        return True

    def _send(self, title, message, image_path, priority, tags) -> bool:
        if self._send_screenshot(title, message, image_path, priority, tags):
            return True
        return self._send_text(title, message, priority, tags)

    def _send_screenshot(self, title, message, image_path, priority, tags) -> bool:
        if not image_path:
            return False
        try:
            with open(image_path, "rb") as f:
                body = f.read()
        except OSError as e:
            logger.debug(f"screenshot not readable ({e}); sending text only")
            return False

        if len(body) > self.max_bytes:
            logger.warning(
                f"screenshot too large for ntfy ({len(body)} > {self.max_bytes} "
                f"bytes); sending text only"
            )
            return False

        headers = {
            "Title": _header(title),
            "Message": _header(message),
            "Filename": _header(image_path.replace("\\", "/").rsplit("/", 1)[-1]),
            "Content-Type": "image/jpeg",
        }
        self._apply_meta(headers, priority, tags)
        return self._urlopen(headers, body, method="PUT")

    def _send_text(self, title, message, priority, tags) -> bool:
        headers = {
            "Title": _header(title),
            "Content-Type": "text/plain; charset=utf-8",
        }
        self._apply_meta(headers, priority, tags)
        return self._urlopen(headers, str(message).encode("utf-8"), method="POST")

    def _apply_meta(self, headers, priority, tags):
        if priority:
            headers["Priority"] = _header(priority)
        if tags:
            headers["Tags"] = _header(tags)
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

    def _urlopen(self, headers, body, method) -> bool:
        url = f"{self.server}/{self.topic}"
        request = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=10) as resp:
                logger.debug(
                    f"notified ntfy topic {self.topic!r} ({method}, "
                    f"HTTP {getattr(resp, 'status', '?')})"
                )
                return True
        except Exception as e:
            logger.warning(f"ntfy notification failed: {e}")
            return False
