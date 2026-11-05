import argparse
import json
import logging
import math
import os
import time

import cv2
import numpy as np

from ai.brain import SYSTEM_INSTRUCTION, infer_target_side
from ai.policy_net import (
    DEFAULT_DATASET_PATH,
    DEFAULT_POLICY_WEIGHTS,
    train_policy_from_dataset,
)
from core.utils import user_path

logger = logging.getLogger(__name__)

DEFAULT_FINETUNE_PATH = "dataset/llm_finetune.jsonl"


class VideoAttackExtractor:
    """
    Extracts human attack demonstrations from Clash of Clans gameplay videos (.mp4/.mkv/.avi)
    or frame sequences to train both:
      1. AttackPolicyNet (Behavioral Cloning for human-like troop tap coordinates & timing)
      2. Local LLM / VLM QLoRA instruction dataset (strategic deploy_order, target_side, duration)
    """

    def __init__(self, vision=None, slot_names: list[str] | None = None):
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
        bar_y1, bar_y2 = int(h * 0.84), int(h * 0.96)
        n_slots = max(1, len(self.slot_names))
        slot_x_start = int(w * 0.16)
        slot_x_end = int(w * 0.56)
        span = max(1, slot_x_end - slot_x_start)

        best_idx = 0
        best_score = -1.0
        for idx in range(n_slots):
            sx1 = slot_x_start + int(span * idx / n_slots)
            sx2 = slot_x_start + int(span * (idx + 1) / n_slots)
            patch_curr = curr_frame[bar_y1:bar_y2, sx1:sx2]
            patch_prev = prev_frame[bar_y1:bar_y2, sx1:sx2]
            if patch_curr.size == 0 or patch_prev.size == 0:
                continue
            diff = float(np.mean(cv2.absdiff(patch_curr, patch_prev)))
            if diff > best_score:
                best_score = diff
                best_idx = idx

        return self.slot_names[best_idx]

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

            # Deduplicate nearby recent taps ( prevents walking troops from registering as new taps )
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

        storages = self._detect_storages(frames[0])
        if not storages:
            # Default representative storage ring if frame has no visible storages
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
        finetune_out: str | None = None,
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
                if len(frames) >= 1200:  # Cap at ~150s @ 8fps
                    break
        finally:
            cap.release()

        demo = self.extract_from_frames(
            frames, fps=effective_fps, source_name=os.path.basename(video_path)
        )
        self.save_demonstration(demo, dataset_out=dataset_out, finetune_out=finetune_out)
        return demo

    def save_demonstration(
        self,
        demo: dict,
        dataset_out: str | None = None,
        finetune_out: str | None = None,
    ) -> tuple[str, str]:
        dataset_out = dataset_out or user_path(DEFAULT_DATASET_PATH)
        finetune_out = finetune_out or user_path(DEFAULT_FINETUNE_PATH)

        os.makedirs(os.path.dirname(dataset_out), exist_ok=True)
        os.makedirs(os.path.dirname(finetune_out), exist_ok=True)

        with open(dataset_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")

        # Also write an instruction-tuning sample for Local LLM QLoRA training (Unsloth / Ollama)
        llm_state = {
            "loot": {"gold": 650000, "elixir": 650000, "delixir": 4500},
            "targets": {"gold": 400000, "elixir": 400000, "delixir": 2000},
            "base": {"storages": demo.get("storages", [])},
            "troop_slots": self.slot_names,
        }
        llm_target = {
            "skip_base": False,
            "deploy_order": demo.get("deploy_order", self.slot_names),
            "surrender_after_s": demo.get("duration_s", 60.0),
            "target_side": demo.get("target_side", "all"),
        }
        finetune_entry = {
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": json.dumps(llm_state, ensure_ascii=False)},
                {
                    "role": "assistant",
                    "content": json.dumps(llm_target, ensure_ascii=False),
                },
            ]
        }
        with open(finetune_out, "a", encoding="utf-8") as f:
            f.write(json.dumps(finetune_entry, ensure_ascii=False) + "\n")

        return dataset_out, finetune_out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract human attack demonstrations from video (.mp4) and train Deep Learning AttackPolicyNet."
    )
    parser.add_argument("--video", help="Path to attack gameplay video (.mp4/.mkv/.avi)")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train AttackPolicyNet on dataset/attack_demos.jsonl after extraction",
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
        print(
            f"Trained AttackPolicyNet on {metrics['device']}: "
            f"{metrics['samples']} tap samples, "
            f"loss {metrics['initial_loss']} -> {metrics['final_loss']} "
            f"(saved to {metrics['weights_path']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
