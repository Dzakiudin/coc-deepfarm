import json
import logging
import os
import time
import uuid

logger = logging.getLogger(__name__)


class RaidRecorder:

    def __init__(self, directory, enabled=True):
        self.directory = directory
        self.enabled = bool(enabled)
        self.state = {}
        self.plan = {}
        self.events = []
        self._start = time.time()
        self._done = False
        if self.enabled:
            try:
                os.makedirs(self.directory, exist_ok=True)
            except Exception as e:
                logger.warning(f"replay: cannot create directory {self.directory}: {e}")
                self.enabled = False

    def start(self, state, plan):
        self.state = state
        self.plan = plan
        self.events = []
        self._start = time.time()
        self._done = False
        return self

    def event(self, event_name, **data):
        self.events.append(
            {"t": round(time.time() - self._start, 3), "event": event_name, **data}
        )

    def finish(self, **outcome):
        if not self.enabled or self._done:
            return None
        self._done = True
        try:
            os.makedirs(self.directory, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            name = f"raid_{stamp}_{uuid.uuid4().hex[:6]}.json"
            path = os.path.join(self.directory, name)
            payload = {
                "state": self.state,
                "plan": self.plan,
                "events": self.events,
                "outcome": outcome,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return path
        except Exception as e:
            logger.warning(f"replay: could not write raid file: {e}")
            return None


class ReplayLogger:

    def __init__(self, directory="logs", enabled=True):
        self.directory = directory
        self.enabled = bool(enabled)

    def start(self, state, plan) -> RaidRecorder:
        recorder = RaidRecorder(self.directory, enabled=self.enabled)
        return recorder.start(state, plan)
