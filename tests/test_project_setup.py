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

            self.assertEqual(loaded, {"controls": {}, "brain": {"name": "coc_ai", "use_ai": True}, "notifications": {"enabled": False}})
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

    def test_flexible_loot_evaluation_and_spatial_feature_extraction(self):
        from ai.brain import (
            evaluate_loot_targets,
            extract_base_features,
            infer_target_side,
        )

        targets = {"gold": 500_000, "elixir": 500_000, "delixir": 5_000}
        jackpot_gold = {"gold": 950_000, "elixir": 300_000, "delixir": 4_000}
        self.assertFalse(evaluate_loot_targets(jackpot_gold, targets, mode="and"))
        self.assertTrue(evaluate_loot_targets(jackpot_gold, targets, mode="or"))
        self.assertTrue(evaluate_loot_targets(jackpot_gold, targets, mode="smart"))

        # Clustered left storages should infer 'left' side
        left_geom = {
            "storages": [
                {"x": 0.25, "y": 0.3, "type": "gold"},
                {"x": 0.30, "y": 0.4, "type": "elixir"},
                {"x": 0.35, "y": 0.5, "type": "gold"},
                {"x": 0.40, "y": 0.6, "type": "elixir"},
            ],
            "dead_base": True,
        }
        self.assertEqual(infer_target_side(left_geom), "left")

        feats = extract_base_features(
            {
                "loot": jackpot_gold,
                "targets": targets,
                "base": left_geom,
                "attack": {"target_mode": "smart"},
            }
        )
        self.assertEqual(len(feats), 16)
        self.assertEqual(feats[4], 1.0)  # dead_base flag
        self.assertEqual(feats[15], 1.0)  # meets_filter flag

    def test_video_learner_and_policy_net_behavioral_cloning_pipeline(self):
        from ai.policy_net import PolicyPlanner, train_policy_from_dataset
        from ai.video_learner import VideoAttackExtractor

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = str(Path(temp_dir) / "attack_demos.jsonl")
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
            extractor.save_demonstration(demo, dataset_out=dataset_path)
            self.assertGreaterEqual(len(demo["taps"]), 1)
            self.assertTrue(Path(dataset_path).is_file())

            metrics = train_policy_from_dataset(
                dataset_path=dataset_path,
                weights_out=weights_path,
                epochs=15,
            )
            self.assertGreaterEqual(metrics["samples"], 1)
            self.assertTrue(Path(weights_path).is_file())

            planner = PolicyPlanner(weights_path=weights_path)
            self.assertTrue(planner.loaded)
            planner.start_raid_episode()
            base_before = {"golds": [(800, 360)], "elixirs": [(600, 320)], "d_elixirs": []}
            pts = planner.predict_points(
                base_before,
                width=1680,
                height=720,
                side="all",
                y_limit=590,
                troop_slot="s_goblin",
            )
            self.assertTrue(len(pts) >= 2)

            # Simulate a failed wave where neither storage was destroyed -> mistake correction
            fb = planner.record_wave_feedback(base_before, base_before, 1680, 720)
            self.assertGreaterEqual(fb["survived"], 2)
            rl_mistake = planner.learn_from_raid_outcome(
                loot_gain={"gold": 20_000, "elixir": 10_000, "delixir": 0},
                base_loot={"gold": 800_000, "elixir": 800_000, "delixir": 0},
                drained=0.05,
                used_slots=["s_goblin", "troop_slot_1"],
            )
            self.assertTrue(rl_mistake["updated"])
            self.assertTrue(rl_mistake["is_mistake"])

            # Simulate a successful raid where storages are destroyed -> reinforcement
            planner.start_raid_episode()
            planner.predict_points(
                base_before,
                width=1680,
                height=720,
                side="all",
                y_limit=590,
                troop_slot="troop_slot_1",
            )
            fb_ok = planner.record_wave_feedback(
                base_before, {"golds": [], "elixirs": [], "d_elixirs": []}, 1680, 720
            )
            self.assertGreaterEqual(fb_ok["destroyed"], 2)
            rl_success = planner.learn_from_raid_outcome(
                loot_gain={"gold": 780_000, "elixir": 760_000, "delixir": 0},
                base_loot={"gold": 800_000, "elixir": 800_000, "delixir": 0},
                drained=0.95,
                used_slots=["troop_slot_1", "s_goblin"],
            )
            self.assertTrue(rl_success["updated"])
            self.assertFalse(rl_success["is_mistake"])

    def test_coc_tactical_brain_neural_planning_and_online_rl(self):
        from ai.brain import CoCTacticalBrain, train_commander_from_data

        with tempfile.TemporaryDirectory() as temp_dir:
            cmd_weights = str(Path(temp_dir) / "coc_commander.pt")
            mem_path = str(Path(temp_dir) / "base_memory.json")
            metrics = train_commander_from_data(
                dataset_path=str(Path(temp_dir) / "nonexistent.jsonl"),
                replay_dir=temp_dir,
                weights_out=cmd_weights,
                epochs=25,
            )
            self.assertGreaterEqual(metrics["samples"], 10)
            self.assertTrue(Path(cmd_weights).is_file())

            brain = CoCTacticalBrain(weights_path=cmd_weights, memory_path=mem_path)
            self.assertTrue(brain.loaded)

            state = {
                "loot": {"gold": 850_000, "elixir": 820_000, "delixir": 5_000},
                "targets": {"gold": 500_000, "elixir": 500_000, "delixir": 3_000},
                "base": {
                    "storages": [
                        {"type": "gold", "x": 0.72, "y": 0.35},
                        {"type": "elixir", "x": 0.78, "y": 0.45},
                        {"type": "gold", "x": 0.75, "y": 0.55},
                        {"type": "elixir", "x": 0.80, "y": 0.62},
                    ],
                    "dead_base": True,
                },
                "troop_slots": ["s_goblin", "troop_slot_1"],
                "attack": {"time_limit": 90.0, "target_mode": "smart"},
            }
            plan = brain.plan(state)
            self.assertFalse(plan["skip_base"])
            self.assertIn(plan["target_side"], ("all", "left", "right"))
            self.assertGreaterEqual(plan["predicted_profit"], 0.5)

            # Test mistake correction on poor raid outcome + Associative Memory AVOID_MISTAKE recall
            rl_bad = brain.learn_from_raid(
                state=state,
                plan_used=plan,
                loot_gain={"gold": 40_000, "elixir": 30_000, "delixir": 0},
                base_loot=state["loot"],
                drained=0.10,
            )
            self.assertTrue(rl_bad["updated"])
            self.assertTrue(rl_bad["is_mistake"])

            plan_after_mistake = brain.plan(state)
            self.assertTrue(plan_after_mistake["memory_match"])
            self.assertEqual(plan_after_mistake["memory_action"], "AVOID_MISTAKE")
            self.assertGreaterEqual(plan_after_mistake["memory_similarity"], 0.95)

            # Test reinforcement on high-loot raid outcome + Associative Memory REINFORCE_WIN recall
            rl_good = brain.learn_from_raid(
                state=state,
                plan_used=plan_after_mistake,
                loot_gain={"gold": 820_000, "elixir": 800_000, "delixir": 4_900},
                base_loot=state["loot"],
                drained=0.95,
            )
            self.assertTrue(rl_good["updated"])
            self.assertFalse(rl_good["is_mistake"])

            plan_after_win = brain.plan(state)
            self.assertTrue(plan_after_win["memory_match"])
            self.assertEqual(plan_after_win["memory_action"], "REINFORCE_WIN")


if __name__ == "__main__":
    unittest.main()
