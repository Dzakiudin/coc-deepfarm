import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import settings
from gui import BotGUI


class SettingsTemplateTests(unittest.TestCase):
    def test_missing_user_settings_are_seeded_from_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            with (
                patch.object(settings, "SETTINGS_PATH", str(settings_path)),
                patch.object(settings, "resource_path", return_value=str(Path(__file__).parents[1] / "settings.example.json")),
            ):
                loaded = settings.load_settings()

            self.assertEqual(loaded, {"controls": {}, "brain": {"use_ai": False}, "notifications": {"enabled": False}})
            self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8")), loaded)

    def test_existing_user_settings_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            settings_path.write_text('{"brain": {"use_ai": true}}', encoding="utf-8")
            with (
                patch.object(settings, "SETTINGS_PATH", str(settings_path)),
                patch.object(settings, "resource_path", side_effect=AssertionError("template must not be read")),
            ):
                loaded = settings.load_settings()

            self.assertEqual(loaded, {"brain": {"use_ai": True}})


class PreviewBufferTests(unittest.TestCase):
    def make_gui(self):
        gui = BotGUI.__new__(BotGUI)
        gui._preview_frame_lock = threading.Lock()
        gui._pending_preview_frame = None
        gui._preview_generation = 0
        gui._rendered_preview_generation = 0
        gui._preview_needs_render = False
        gui._preview_stop = threading.Event()
        gui.latest_frame = None
        gui._last_telemetry_at = None
        gui.root = type("Root", (), {"after": lambda *_args: None})()
        gui._render_preview = lambda: None
        return gui

    def test_preview_keeps_only_latest_frame_and_does_not_render_twice(self):
        gui = self.make_gui()
        rendered = []
        gui._render_preview = lambda: rendered.append(gui.latest_frame)
        first_frame = np.zeros((2, 2, 3), dtype=np.uint8)
        latest_frame = np.ones((2, 2, 3), dtype=np.uint8)

        gui._queue_preview_frame(first_frame)
        gui._queue_preview_frame(latest_frame)
        gui._refresh_preview()
        gui._refresh_preview()

        self.assertEqual(len(rendered), 1)
        np.testing.assert_array_equal(rendered[0], latest_frame)


if __name__ == "__main__":
    unittest.main()
