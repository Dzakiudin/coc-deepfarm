import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import settings
from ui.gui import BotGUI


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


class FarmingEnhancementsTests(unittest.TestCase):
    def _make_bot(self):
        import queue
        from core.bot import CocBot
        from core.config import AppConfig

        cfg = AppConfig()
        win = type(
            "Win",
            (),
            {
                "win_left": 0,
                "win_top": 0,
                "win_width": 1680,
                "win_height": 720,
                "last_capture_ms": 5.0,
                "video_stream_active": True,
                "capture_window": lambda self_w: np.zeros((720, 1680, 3), dtype=np.uint8),
            },
        )()
        vis = type(
            "Vis",
            (),
            {
                "last_inference_ms": 10.0,
                "last_ocr_ms": 5.0,
                "detect_with_details": lambda self_v, frame, targets, conf=None: {},
            },
        )()
        tapped = []
        clicked = []
        auto = type(
            "Auto",
            (),
            {
                "click": lambda self_a, x, y=None: clicked.append((x, y)),
                "tap_many": lambda self_a, pts: tapped.extend(pts),
                "hold": lambda self_a, x, y=None, seconds=0.6: None,
            },
        )()
        controls = {
            "surr": {"x": 0.05417, "y": 0.74583},       # (91, 537)
            "s_goblin": {"x": 0.20179, "y": 0.90417},   # (339, 651)
            "troop_slot_1": {"x": 0.27143, "y": 0.88889},  # (456, 640)
        }
        bot = CocBot(
            cfg,
            win,
            vis,
            auto,
            queue.Queue(),
            controls=controls,
            troops={"slots": ["s_goblin", "troop_slot_1"], "deploy_mode": "target"},
            waits={"attack_wave_settle": 0.0, "attack_settle": 0.0},
        )
        return bot, clicked, tapped

    def test_bottom_base_storages_are_not_blind_spotted_and_surrender_is_protected(self):
        bot, _, _ = self._make_bot()
        limit = bot._deploy_y_limit()
        # Previously limit was 537 - 100 = 437; now it reaches above troop bar (640 - 45 = 595)
        self.assertGreaterEqual(limit, 580)

        # Storage at y=510 (bottom half of base) should now produce rows and target points
        base = {"golds": [(850, 510)], "elixirs": [], "d_elixirs": []}
        rows = bot._attack_rows(base, "all")
        self.assertIn(510, rows)
        targets = bot._deploy_targets(base, "all")
        self.assertTrue(len(targets) > 0)

        # Surrender button zone at bottom-left (91, 537) must remain protected
        self.assertTrue(bot._is_in_surrender_zone(95, 535))
        self.assertFalse(bot._is_in_surrender_zone(850, 535))

    def test_validate_plan_preserves_left_and_right_target_side(self):
        bot, _, _ = self._make_bot()
        validated = bot._validate_plan(
            {
                "skip_base": False,
                "deploy_order": ["s_goblin"],
                "surrender_after_s": 60,
                "target_side": "left",
            },
            {},
        )
        self.assertEqual(validated["target_side"], "left")

    def test_empty_troop_slots_are_skipped_on_subsequent_waves(self):
        bot, _, _ = self._make_bot()
        # Create a colorful frame for wave 1 around both troop slots
        colorful = np.zeros((720, 1680, 3), dtype=np.uint8)
        colorful[:, :] = (0, 180, 255)  # High saturation orange in BGR
        bot.window.capture_window = lambda: colorful

        baseline = {}
        active_w1 = bot._filter_active_slots(["s_goblin", "troop_slot_1"], baseline, wave=1)
        self.assertEqual(active_w1, ["s_goblin", "troop_slot_1"])
        self.assertIn("s_goblin", baseline)

        # On wave 2, make s_goblin (339, 651) grayscale (depleted) while troop_slot_1 stays colorful
        gray_patch_frame = colorful.copy()
        gray_patch_frame[630:670, 315:365] = (120, 120, 120)
        bot.window.capture_window = lambda: gray_patch_frame

        active_w2 = bot._filter_active_slots(["s_goblin", "troop_slot_1"], baseline, wave=2)
        self.assertEqual(active_w2, ["troop_slot_1"])

    def test_flexible_loot_evaluation_and_ai_quota_prefilter(self):
        from ai.brain import (
            evaluate_loot_targets,
            infer_target_side,
            normalize_model_name,
            should_consult_ai,
        )

        self.assertEqual(normalize_model_name("gemini-3.5-flash"), "gemini-2.5-flash-lite")
        self.assertEqual(normalize_model_name("gemini-3.8-flash"), "gemini-2.5-flash")

        targets = {"gold": 500_000, "elixir": 500_000, "delixir": 5_000}
        jackpot_gold = {"gold": 950_000, "elixir": 300_000, "delixir": 4_000}
        self.assertFalse(evaluate_loot_targets(jackpot_gold, targets, mode="and"))
        self.assertTrue(evaluate_loot_targets(jackpot_gold, targets, mode="or"))
        self.assertTrue(evaluate_loot_targets(jackpot_gold, targets, mode="smart"))

        # Low-loot base should not burn Gemini API quota
        low_state = {
            "loot": {"gold": 80_000, "elixir": 90_000, "delixir": 100},
            "targets": targets,
            "base": {"dead_base": False},
            "attack": {"target_mode": "smart"},
        }
        self.assertFalse(should_consult_ai(low_state))

        # Clustered left storages should infer 'left' side
        left_geom = {
            "storages": [
                {"x": 0.25, "y": 0.3},
                {"x": 0.30, "y": 0.4},
                {"x": 0.35, "y": 0.5},
                {"x": 0.40, "y": 0.6},
            ]
        }
        self.assertEqual(infer_target_side(left_geom), "left")

    def test_video_learner_and_policy_net_behavioral_cloning_pipeline(self):
        from ai.policy_net import PolicyPlanner, train_policy_from_dataset
        from ai.video_learner import VideoAttackExtractor

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = str(Path(temp_dir) / "attack_demos.jsonl")
            llm_path = str(Path(temp_dir) / "llm_finetune.jsonl")
            weights_path = str(Path(temp_dir) / "attack_policy.pt")

            extractor = VideoAttackExtractor(
                vision=None,
                slot_names=["s_goblin", "troop_slot_1"],
            )

            f1 = np.zeros((360, 640, 3), dtype=np.uint8)
            f2 = np.zeros((360, 640, 3), dtype=np.uint8)
            # Simulate a troop spawn burst on the battlefield around (300, 180)
            f2[165:190, 285:310] = 255

            demo = extractor.extract_from_frames([f1, f2], fps=4.0)
            extractor.save_demonstration(
                demo, dataset_out=dataset_path, finetune_out=llm_path
            )
            self.assertGreaterEqual(len(demo["taps"]), 1)
            self.assertTrue(Path(dataset_path).is_file())
            self.assertTrue(Path(llm_path).is_file())

            metrics = train_policy_from_dataset(
                dataset_path=dataset_path,
                weights_out=weights_path,
                epochs=15,
            )
            self.assertGreaterEqual(metrics["samples"], 1)
            self.assertTrue(Path(weights_path).is_file())

            planner = PolicyPlanner(weights_path=weights_path)
            self.assertTrue(planner.loaded)
            pts = planner.predict_points(
                {"golds": [(800, 360)], "elixirs": [(600, 320)], "d_elixirs": []},
                width=1680,
                height=720,
                side="all",
                y_limit=590,
            )
            self.assertTrue(len(pts) >= 2)

    def test_local_llm_brain_parses_multimodal_json_response(self):
        from ai.brain import LocalLLMBrain

        captured_requests = []

        class _FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "response": json.dumps(
                            {
                                "skip_base": False,
                                "deploy_order": ["s_goblin", "troop_slot_1"],
                                "surrender_after_s": 55,
                                "target_side": "right",
                            }
                        )
                    }
                ).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            captured_requests.append(req)
            return _FakeResp()

        brain = LocalLLMBrain(
            model="qwen2.5vl:3b",
            vision=True,
        )
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            plan = brain.plan(
                {
                    "loot": {"gold": 700_000, "elixir": 700_000, "delixir": 4_000},
                    "targets": {"gold": 500_000, "elixir": 500_000, "delixir": 3_000},
                    "base": {"storages": [{"type": "gold", "x": 0.72, "y": 0.45}]},
                    "troop_slots": ["s_goblin", "troop_slot_1"],
                    "attack": {"time_limit": 60.0},
                },
                frame=dummy_frame,
            )
        self.assertEqual(plan.get("target_side"), "right")
        self.assertFalse(plan.get("skip_base"))
        self.assertEqual(len(captured_requests), 1)
        sent = json.loads(captured_requests[0].data.decode("utf-8"))
        self.assertEqual(sent["model"], "qwen2.5vl:3b")
        self.assertIn("images", sent)


if __name__ == "__main__":
    unittest.main()
