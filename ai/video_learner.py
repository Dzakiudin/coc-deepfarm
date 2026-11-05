import argparse
import json
import logging
import math
import os
import time

import cv2
import numpy as np

from ai.brain import infer_target_side, train_commander_from_data
from ai.policy_net import (
    DEFAULT_DATASET_PATH,
    train_policy_from_dataset,
)
from core.utils import user_path

logger = logging.getLogger(__name__)


class VideoAttackExtractor:
    """
    Extracts human attack demonstrations from Clash of Clans gameplay videos (.mp4/.mkv/.avi)
    or frame sequences to train both:
      1. AttackPolicyNet (Micro-tactical troop tap coordinates, perimeter projection & timing)
      2. CoCCommanderNet (Macro-strategic side selection, troop sequence & dynamic timer)
    """

    def __init__(
        self,
        vision=None,
        slot_names: list[str] | None = None,
        controls: dict | None = None,
    ):
        self.vision = vision
        self.slot_names = list(
            slot_names
            or [
                "s_goblin",
                "troop_slot_1",
                "troop_slot_2",
                "troop_slot_3",
                "troop_slot_4",
                "troop_slot_5",
            ]
        )
        self.controls = dict(controls or {})
        self._active_slot = self.slot_names[0] if self.slot_names else "s_goblin"

    def _detect_storages(self, frame) -> list[dict]:
        h, w = frame.shape[:2]
        storages = []
        if self.vision is not None:
            try:
                details = self.vision.detect_with_details(
                    frame, ["golds", "elixirs", "d_elixirs"], conf=0.30
                )
                for cls_key, label in (
                    ("golds", "gold"),
                    ("elixirs", "elixir"),
                    ("d_elixirs", "delixir"),
                ):
                    for item in details.get(cls_key, []):
                        cx, cy = item["center"]
                        storages.append(
                            {
                                "cls": label,
                                "x": round(float(cx) / w, 4),
                                "y": round(float(cy) / h, 4),
                            }
                        )
            except Exception as e:
                logger.debug("YOLO storage detection on video frame failed: %s", e)

        if storages:
            return storages

        # Fallback heuristic when YOLO is not loaded: detect high-saturation gold/elixir patches
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        y1, y2 = int(h * 0.12), int(h * 0.78)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        roi = hsv[y1:y2, x1:x2]

        gold_mask = cv2.inRange(roi, (18, 130, 150), (32, 255, 255))
        elixir_mask = cv2.inRange(roi, (140, 110, 140), (172, 255, 255))

        for mask, label in ((gold_mask, "gold"), (elixir_mask, "elixir")):
            cnts, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for c in cnts:
                area = cv2.contourArea(c)
                if 180 <= area <= 12000:
                    bx, by, bw, bh = cv2.boundingRect(c)
                    cx = x1 + bx + bw / 2.0
                    cy = y1 + by + bh / 2.0
                    storages.append(
                        {
                            "cls": label,
                            "x": round(cx / w, 4),
                            "y": round(cy / h, 4),
                        }
                    )
                    if len(storages) >= 16:
                        break
        return storages[:16]

    def _detect_active_slot(self, prev_frame, curr_frame) -> str:
        h, w = curr_frame.shape[:2]
        n_slots = max(1, len(self.slot_names))
        best_slot = None
        best_score = 0.0

        for idx, slot_name in enumerate(self.slot_names):
            norm = self.controls.get(slot_name)
            if isinstance(norm, dict) and "x" in norm and "y" in norm:
                cx = int(float(norm["x"]) * w)
                cy = int(float(norm["y"]) * h)
                sx1, sx2 = max(0, cx - int(w * 0.025)), min(w, cx + int(w * 0.025))
                sy1, sy2 = max(0, cy - int(h * 0.055)), min(h, cy + int(h * 0.045))
            else:
                sy1, sy2 = int(h * 0.84), int(h * 0.96)
                slot_x_start, slot_x_end = int(w * 0.16), int(w * 0.56)
                span = max(1, slot_x_end - slot_x_start)
                sx1 = slot_x_start + int(span * idx / n_slots)
                sx2 = slot_x_start + int(span * (idx + 1) / n_slots)

            patch_curr = curr_frame[sy1:sy2, sx1:sx2]
            patch_prev = prev_frame[sy1:sy2, sx1:sx2]
            if patch_curr.size == 0 or patch_prev.size == 0:
                continue
            # Ignore grayscale (depleted) slots
            hsv = cv2.cvtColor(patch_curr, cv2.COLOR_BGR2HSV)
            if float(hsv[:, :, 1].mean()) < 26.0:
                continue
            diff = float(np.mean(cv2.absdiff(patch_curr, patch_prev)))
            if diff > best_score:
                best_score = diff
                best_slot = slot_name

        if best_slot is not None and best_score >= 6.5:
            self._active_slot = best_slot
        return self._active_slot

    @staticmethod
    def _detect_spawn_taps(prev_frame, curr_frame, existing_taps, t_sec: float) -> list[dict]:
        """
        Detect sudden localized troop deployment flashes/rings on the battlefield between frames.
        """
        h, w = curr_frame.shape[:2]
        play_y1, play_y2 = int(h * 0.08), int(h * 0.80)
        play_x1, play_x2 = int(w * 0.05), int(w * 0.94)

        prev_gray = cv2.cvtColor(
            prev_frame[play_y1:play_y2, play_x1:play_x2], cv2.COLOR_BGR2GRAY
        )
        curr_gray = cv2.cvtColor(
            curr_frame[play_y1:play_y2, play_x1:play_x2], cv2.COLOR_BGR2GRAY
        )

        diff = cv2.absdiff(curr_gray, prev_gray)
        _, thresh = cv2.threshold(diff, 38, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        new_taps = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (45 <= area <= 4500):
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > 110 or bh > 110:
                continue
            nx = round((play_x1 + bx + bw / 2.0) / w, 4)
            ny = round((play_y1 + by + bh / 2.0) / h, 4)

            if any(
                abs(t_sec - prev["t"]) < 1.2
                and math.hypot(nx - prev["x"], ny - prev["y"]) < 0.065
                for prev in existing_taps[-25:]
            ):
                continue
            new_taps.append({"t": round(t_sec, 3), "x": nx, "y": ny})
            if len(new_taps) >= 6:
                break
        return new_taps

    def extract_from_frames(
        self, frames: list, fps: float = 8.0, source_name: str = "frames"
    ) -> dict:
        if len(frames) < 2:
            raise ValueError("Need at least 2 frames to extract attack trajectory")

        storages = []
        probe_indices = sorted(
            {
                min(len(frames) - 1, int(len(frames) * frac))
                for frac in (0.0, 0.05, 0.12, 0.20, 0.30, 0.40)
            }
        )
        for p_idx in probe_indices:
            candidate = self._detect_storages(frames[p_idx])
            if len(candidate) > len(storages):
                storages = candidate
            if len(storages) >= 10:
                break

        if not storages:
            storages = [
                {"cls": "gold", "x": 0.35, "y": 0.35},
                {"cls": "elixir", "x": 0.65, "y": 0.35},
                {"cls": "gold", "x": 0.35, "y": 0.60},
                {"cls": "elixir", "x": 0.65, "y": 0.60},
            ]

        taps = []
        deploy_order = []
        prev_frame = frames[0]
        last_tap_t = 0.0

        for idx in range(1, len(frames)):
            curr_frame = frames[idx]
            t_sec = idx / max(1.0, float(fps))
            slot = self._detect_active_slot(prev_frame, curr_frame)
            spawn_pts = self._detect_spawn_taps(prev_frame, curr_frame, taps, t_sec)
            for pt in spawn_pts:
                if slot not in deploy_order:
                    deploy_order.append(slot)
                nearest = min(
                    storages,
                    key=lambda s: math.hypot(pt["x"] - s["x"], pt["y"] - s["y"]),
                )
                dt = max(0.02, min(0.50, pt["t"] - last_tap_t)) if last_tap_t > 0 else 0.08
                last_tap_t = pt["t"]
                taps.append(
                    {
                        "t": pt["t"],
                        "dt": round(dt, 4),
                        "x": pt["x"],
                        "y": pt["y"],
                        "slot": slot,
                        "nearest_store_x": nearest["x"],
                        "nearest_store_y": nearest["y"],
                        "nearest_store_cls": nearest["cls"],
                    }
                )
            prev_frame = curr_frame

        if not deploy_order:
            deploy_order = list(self.slot_names)

        duration_s = round(len(frames) / max(1.0, float(fps)), 1)
        inferred_side = infer_target_side(
            {"storages": [{"x": t["x"], "y": t["y"]} for t in taps]}
            if len(taps) >= 4
            else {"storages": storages}
        )

        return {
            "source": source_name,
            "timestamp": time.time(),
            "duration_s": max(15.0, min(180.0, duration_s)),
            "storages": storages,
            "deploy_order": deploy_order,
            "target_side": inferred_side,
            "taps": taps,
        }

    def extract_from_video(
        self,
        video_path: str,
        sample_fps: float = 8.0,
        dataset_out: str | None = None,
    ) -> dict:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        if native_fps <= 1.0:
            native_fps = 30.0
        step = max(1, int(round(native_fps / max(1.0, sample_fps))))
        effective_fps = native_fps / step

        frames = []
        frame_idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if frame_idx % step == 0:
                    frames.append(frame)
                frame_idx += 1
                if len(frames) >= 1200:
                    break
        finally:
            cap.release()

        demo = self.extract_from_frames(
            frames, fps=effective_fps, source_name=os.path.basename(video_path)
        )
        self.save_demonstration(demo, dataset_out=dataset_out)
        return demo

    def save_demonstration(
        self,
        demo: dict,
        dataset_out: str | None = None,
    ) -> str:
        dataset_out = dataset_out or user_path(DEFAULT_DATASET_PATH)
        os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
        with open(dataset_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
        return dataset_out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract human attack demonstrations from video (.mp4) and train CoC Deep Learning Networks."
    )
    parser.add_argument("--video", help="Path to attack gameplay video (.mp4/.mkv/.avi)")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train AttackPolicyNet & CoCCommanderNet on dataset/attack_demos.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs")
    args = parser.parse_args(argv)

    if args.video:
        extractor = VideoAttackExtractor()
        demo = extractor.extract_from_video(args.video)
        print(
            f"Extracted {len(demo.get('taps', []))} human taps and "
            f"{len(demo.get('storages', []))} storages from {demo['source']} "
            f"(side={demo['target_side']}, order={demo['deploy_order']})"
        )

    if args.train or args.video:
        metrics = train_policy_from_dataset(epochs=args.epochs)
        cmd_metrics = train_commander_from_data(epochs=args.epochs)
        print(
            f"Trained AttackPolicyNet on {metrics['device']}: "
            f"{metrics['samples']} tap samples, "
            f"loss {metrics['initial_loss']} -> {metrics['final_loss']} "
            f"(saved to {metrics['weights_path']})"
        )
        print(
            f"Trained CoCCommanderNet on {cmd_metrics['device']}: "
            f"{cmd_metrics['samples']} strategic samples, "
            f"loss {cmd_metrics['initial_loss']} -> {cmd_metrics['final_loss']} "
            f"(saved to {cmd_metrics['weights_path']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
