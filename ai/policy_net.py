import json
import logging
import math
import os
from collections import Counter, defaultdict

import numpy as np

from core.utils import user_path

logger = logging.getLogger(__name__)

DEFAULT_POLICY_WEIGHTS = "models/policy_net.pt"
DEFAULT_DATASET_PATH = "dataset/attack_demos.jsonl"
DEFAULT_SELF_LEARN_PATH = "dataset/self_learned_raids.jsonl"


def _get_torch():
    import torch
    import torch.nn as nn

    return torch, nn


class PolicyPlanner:
    """
    Deep Learning Behavioral Cloning Policy + Learned Strategic Profile for Clash of Clans.
    Learns both:
      1. Macro attack habits from user videos: troop slot deploy_order, per-slot tap budget,
         preferred attack side, and battle duration.
      2. Micro troop placement neural network (AttackPolicyNet on CUDA RTX 4060):
         predicts human-like perimeter/flank deployment coordinates per troop slot
         relative to enemy storages while avoiding the inner red no-deploy zone.
    """

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        self.weights_path = weights_path or user_path(DEFAULT_POLICY_WEIGHTS)
        self.device = device
        self.model = None
        self.strategy_profile: dict = {}
        self.loaded = False
        self._episode_records: list[dict] = []
        self._prev_wave_stores: list[tuple[float, float, str]] = []
        self.reload()

    @staticmethod
    def _build_network(torch, nn):
        class AttackPolicyNet(nn.Module):
            """
            Input features (9 dims):
              [store_x, store_y, out_vx, out_vy, dist_to_center,
               is_gold, is_elixir, ring_slot_norm, troop_slot_norm]
            Output (3 dims):
              [offset_dx, offset_dy, inter_tap_delay_s]
            """

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(9, 64),
                    nn.SiLU(),
                    nn.LayerNorm(64),
                    nn.Linear(64, 64),
                    nn.SiLU(),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                    nn.Linear(32, 3),
                    nn.Tanh(),
                )

            def forward(self, x):
                raw = self.net(x)
                dx = raw[:, 0:1] * 0.22
                dy = raw[:, 1:2] * 0.22
                delay = (raw[:, 2:3] + 1.0) * 0.12 + 0.02
                return torch.cat([dx, dy, delay], dim=1)

        return AttackPolicyNet()

    def reload(self) -> bool:
        if not os.path.isfile(self.weights_path):
            self.loaded = False
            self.strategy_profile = {}
            return False
        try:
            torch, nn = _get_torch()
            if self.device is None:
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            state = torch.load(
                self.weights_path, map_location=self.device, weights_only=True
            )
            net = self._build_network(torch, nn).to(self.device)
            net.load_state_dict(state["model_state"])
            net.eval()
            self.model = net
            self.strategy_profile = dict(state.get("strategy_profile") or {})
            self.loaded = True
            logger.info(
                "Loaded Deep Learning AttackPolicyNet from %s on %s (order=%s)",
                self.weights_path,
                self.device,
                self.strategy_profile.get("deploy_order"),
            )
            return True
        except Exception as e:
            logger.warning("Could not load AttackPolicyNet weights: %s", e)
            self.loaded = False
            self.strategy_profile = {}
            return False

    @staticmethod
    def _slot_index_norm(slot_name: str | None) -> float:
        order = [
            "s_goblin",
            "troop_slot_1",
            "troop_slot_2",
            "troop_slot_3",
            "troop_slot_4",
            "troop_slot_5",
        ]
        if slot_name in order:
            return float(order.index(slot_name)) / float(len(order) - 1)
        return 0.0

    @staticmethod
    def featurize_store(
        sx: float,
        sy: float,
        cls_name: str,
        slot_k: int,
        max_k: int = 4,
        troop_slot: str | None = None,
    ) -> list[float]:
        # Outward vector from base center (0.50, 0.45) through storage (sx, sy)
        out_vx = sx - 0.50
        out_vy = sy - 0.45
        dist = math.hypot(out_vx, out_vy)
        if dist > 1e-4:
            out_vx /= dist
            out_vy /= dist
        else:
            out_vx, out_vy = 0.0, 1.0
        is_gold = 1.0 if "gold" in str(cls_name).lower() else 0.0
        is_elixir = (
            1.0
            if "elixir" in str(cls_name).lower() and "d_" not in str(cls_name).lower()
            else 0.0
        )
        k_norm = float(slot_k) / max(1, max_k - 1) if max_k > 1 else 0.0
        t_norm = PolicyPlanner._slot_index_norm(troop_slot)
        return [sx, sy, out_vx, out_vy, dist, is_gold, is_elixir, k_norm, t_norm]

    def get_learned_deploy_order(self, available_slots: list[str]) -> list[str]:
        """Return the troop slot order learned from the user's gameplay videos."""
        learned = self.strategy_profile.get("deploy_order") or []
        result = [s for s in learned if s in available_slots]
        for s in available_slots:
            if s not in result:
                result.append(s)
        return result

    def get_learned_surrender_s(self, default_s: float = 90.0) -> float:
        val = self.strategy_profile.get("surrender_after_s")
        try:
            return max(30.0, min(180.0, float(val)))
        except (TypeError, ValueError):
            return default_s

    def get_slot_tap_budget(self, slot_name: str, total_points: int, wave: int = 1) -> int:
        """
        Determine how many taps a specific troop slot should execute in this wave
        based on the user's video demonstrations.
        """
        weights = self.strategy_profile.get("slot_weights") or {}
        if not weights or slot_name not in weights:
            return total_points
        w = float(weights.get(slot_name, 0.2))
        max_w = max(float(v) for v in weights.values()) if weights else 1.0
        rel = w / max(0.05, max_w)
        # Primary farming troop slots get full target coverage; secondary/support slots get proportional taps
        if rel >= 0.55:
            return total_points
        budget = max(4, int(round(total_points * max(0.25, rel))))
        if wave > 1:
            budget = max(4, budget // 2)
        return min(total_points, budget)

    def predict_points(
        self,
        base: dict,
        width: int,
        height: int,
        side: str = "all",
        y_limit: int | None = None,
        surrender_zone_fn=None,
        per_store: int = 4,
        max_points: int = 48,
        troop_slot: str | None = None,
    ) -> list[tuple[int, int]]:
        """
        Use the trained neural network + perimeter projection to predict human-like
        deployment coordinates on the deployable grass outside enemy storages.
        """
        if not self.loaded or self.model is None:
            return []

        torch, _ = _get_torch()
        w = float(max(1, width))
        h = float(max(1, height))
        limit = y_limit if y_limit is not None else int(h * 0.82)

        x_bounds = {"all": (0.0, 1.0), "left": (0.0, 0.5), "right": (0.5, 1.0)}.get(
            side, (0.0, 1.0)
        )

        features = []
        anchors = []
        store_list = []
        for cls_key in ("golds", "elixirs", "d_elixirs"):
            for pt in base.get(cls_key, []):
                try:
                    cx, cy = float(pt[0]), float(pt[1])
                except (TypeError, IndexError, ValueError):
                    continue
                nx, ny = cx / w, cy / h
                store_list.append((nx, ny))
                if not (x_bounds[0] <= nx <= x_bounds[1]):
                    continue
                for k in range(per_store):
                    features.append(
                        self.featurize_store(
                            nx, ny, cls_key, k, per_store, troop_slot=troop_slot
                        )
                    )
                    anchors.append((nx, ny, k))

        if not features:
            return []

        with torch.no_grad():
            tensor_in = torch.tensor(
                features, dtype=torch.float32, device=self.device
            )
            preds = self.model(tensor_in).cpu().numpy()

        min_center_r = float(self.strategy_profile.get("min_deploy_radius", 0.25))
        explore_sigma = float(self.strategy_profile.get("explore_sigma", 0.025))
        rng = np.random.default_rng()
        points = []
        for (nx, ny, k), feat, pred in zip(anchors, features, preds):
            dx, dy, delay = float(pred[0]), float(pred[1]), float(pred[2])
            if explore_sigma > 0.0:
                dx = float(np.clip(dx + rng.normal(0.0, explore_sigma), -0.22, 0.22))
                dy = float(np.clip(dy + rng.normal(0.0, explore_sigma), -0.22, 0.22))

            # Outward radial direction from base center (0.50, 0.45) through storage (nx, ny)
            vx = nx - 0.50
            vy = ny - 0.45
            dist_c = math.hypot(vx, vy)
            if dist_c > 1e-4:
                ux, uy = vx / dist_c, vy / dist_c
            else:
                ux, uy = 0.0, 1.0

            # Blend learned neural offset with outward perimeter push so taps never land inside base walls
            outward_push = max(0.07, min_center_r - dist_c + 0.06)
            base_tx = nx + ux * outward_push + dx * 0.45
            base_ty = ny + uy * outward_push + dy * 0.45

            # Fan points along the tangent arc around the storage flank
            tangent_x, tangent_y = -uy, ux
            arc_offset = (k - (per_store - 1) / 2.0) * 0.038
            tx = base_tx + tangent_x * arc_offset
            ty = base_ty + tangent_y * arc_offset

            # Ensure point is outside the inner no-deploy core around (0.50, 0.45)
            rc = math.hypot(tx - 0.50, ty - 0.45)
            if rc < min_center_r and rc > 1e-4:
                scale = min_center_r / rc
                tx = 0.50 + (tx - 0.50) * scale
                ty = 0.45 + (ty - 0.45) * scale

            px = int(round(tx * w))
            py = int(round(ty * h))
            px = max(25, min(int(w) - 105, px))
            py = max(25, min(limit - 5, py))
            if surrender_zone_fn is not None and surrender_zone_fn(px, py):
                continue
            # Keep clear of storage centers themselves (building footprints)
            if any(
                (px - sx * w) ** 2 + (py - sy * h) ** 2 < 48 ** 2
                for sx, sy in store_list
            ):
                continue
            if any((px - ox) ** 2 + (py - oy) ** 2 < 14 ** 2 for ox, oy in points):
                continue
            points.append((px, py))
            self._episode_records.append(
                {
                    "feat": feat,
                    "store_xy": (nx, ny),
                    "applied_dx": max(-0.22, min(0.22, (px / w) - nx)),
                    "applied_dy": max(-0.22, min(0.22, (py / h) - ny)),
                    "pred_dx": dx,
                    "pred_dy": dy,
                    "delay": delay,
                    "slot": troop_slot or "s_goblin",
                    "local_reward": 0.0,
                }
            )
            if len(points) >= max_points:
                break

        # Sort points in a smooth human-like sweep around the perimeter (by angle from center)
        points.sort(key=lambda p: math.atan2(p[1] - h * 0.45, p[0] - w * 0.50))
        return points

    def start_raid_episode(self):
        """Reset the trajectory buffer at the start of a new raid."""
        self._episode_records = []
        self._prev_wave_stores = []

    def record_wave_feedback(
        self, prev_base: dict, curr_base: dict, width: int, height: int
    ) -> dict:
        """
        Compare storages before and after a wave to see which storages were destroyed
        and which ones survived (mistake / blocked path / red-zone tap).
        Assigns per-storage local rewards to the episode buffer.
        """
        w = float(max(1, width))
        h = float(max(1, height))

        def _extract_norm(b):
            out = []
            for cls_key in ("golds", "elixirs", "d_elixirs"):
                for pt in b.get(cls_key, []):
                    try:
                        out.append((float(pt[0]) / w, float(pt[1]) / h))
                    except (TypeError, IndexError, ValueError):
                        continue
            return out

        before = _extract_norm(prev_base)
        after = _extract_norm(curr_base)
        if not before:
            return {"destroyed": 0, "survived": 0}

        destroyed_count = 0
        survived_count = 0
        for rec in self._episode_records:
            if rec.get("local_reward", 0.0) != 0.0:
                continue
            sx, sy = rec["store_xy"]
            still_alive = any(
                math.hypot(sx - ax, sy - ay) < 0.055 for ax, ay in after
            )
            if still_alive:
                rec["local_reward"] = -0.5  # Storage was not destroyed by this wave
                survived_count += 1
            else:
                rec["local_reward"] = +1.0  # Storage was successfully destroyed!
                destroyed_count += 1

        return {"destroyed": destroyed_count, "survived": survived_count}

    def learn_from_raid_outcome(
        self,
        *,
        loot_gain: dict,
        base_loot: dict,
        drained: float | None,
        used_slots: list[str] | None = None,
        lr: float = 2e-3,
        steps: int = 18,
    ) -> dict:
        """
        Online Reinforcement Learning (Reward-Weighted Regression + Mistake Correction)
        executed automatically after every raid on the GPU (RTX 4060).
        - When a raid succeeds (high loot / high drain), reinforces the successful offsets.
        - When a raid fails or underperforms (mistake: low loot / low drain / surviving storages),
          computes a counterfactual perimeter correction target and updates AttackPolicyNet
          weights + strategy_profile so the bot does not repeat the same mistake.
        """
        if not self.loaded or self.model is None or not self._episode_records:
            return {"updated": False, "reason": "no_episode_data"}

        torch, nn = _get_torch()

        avail_ge = max(
            1,
            int(base_loot.get("gold", 0) or 0) + int(base_loot.get("elixir", 0) or 0),
        )
        gained_ge = int(loot_gain.get("gold", 0) or 0) + int(
            loot_gain.get("elixir", 0) or 0
        )
        loot_ratio = max(0.0, min(1.0, gained_ge / float(avail_ge)))
        drain_val = (
            max(0.0, min(1.0, float(drained)))
            if isinstance(drained, (int, float))
            else loot_ratio
        )
        global_reward = 0.60 * loot_ratio + 0.40 * drain_val
        is_mistake = global_reward < 0.48

        X_rows = []
        Y_targets = []
        sample_weights = []

        for rec in self._episode_records:
            feat = rec["feat"]
            sx, sy = rec["store_xy"]
            adx, ady = float(rec["applied_dx"]), float(rec["applied_dy"])
            local_r = float(rec.get("local_reward", 0.0))

            # Combined advantage for this specific tap action
            advantage = (global_reward - 0.50) + 0.35 * local_r

            if advantage >= 0.0:
                # Positive outcome: reinforce the applied offset that worked
                target_dx = adx
                target_dy = ady
                weight = max(0.25, min(2.0, 0.5 + advantage * 1.5))
            else:
                # Mistake / failed tap: push offset further outward toward the safe perimeter
                # and rotate slightly along the flank so next raid avoids the blocked/red zone
                vx, vy = sx - 0.50, sy - 0.45
                dist_c = math.hypot(vx, vy)
                ux, uy = (vx / dist_c, vy / dist_c) if dist_c > 1e-4 else (0.0, 1.0)
                tx, ty = -uy, ux
                target_dx = max(-0.22, min(0.22, ux * 0.14 + tx * 0.05 - adx * 0.25))
                target_dy = max(-0.22, min(0.22, uy * 0.14 + ty * 0.05 - ady * 0.25))
                weight = max(0.40, min(2.0, abs(advantage) * 1.6))

            X_rows.append(feat)
            Y_targets.append([target_dx, target_dy, float(rec.get("delay", 0.08))])
            sample_weights.append(weight)

        self.model.train()
        X_t = torch.tensor(X_rows, dtype=torch.float32, device=self.device)
        Y_t = torch.tensor(Y_targets, dtype=torch.float32, device=self.device)
        W_t = torch.tensor(
            sample_weights, dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        initial_loss = 0.0
        final_loss = 0.0
        for step_i in range(max(1, int(steps))):
            optimizer.zero_grad()
            pred = self.model(X_t)
            per_elem = torch.nn.functional.smooth_l1_loss(pred, Y_t, reduction="none")
            loss = (per_elem * W_t).mean()
            if step_i == 0:
                initial_loss = float(loss.item())
            loss.backward()
            optimizer.step()
            final_loss = float(loss.item())
        self.model.eval()

        # Adapt strategy_profile parameters based on whether the raid was a mistake or success
        min_r = float(self.strategy_profile.get("min_deploy_radius", 0.26))
        sigma = float(self.strategy_profile.get("explore_sigma", 0.025))
        raids_learned = int(self.strategy_profile.get("raids_self_learned", 0)) + 1

        if is_mistake:
            # If too many troops failed to deploy or died early, expand min_deploy_radius outward
            min_r = round(min(0.33, min_r + 0.008), 4)
            sigma = round(min(0.055, sigma + 0.006), 4)
            # Rotate non-goblin lead troop slot if current lead slot underperformed
            order = list(self.strategy_profile.get("deploy_order") or [])
            if len(order) >= 2:
                order = order[1:] + [order[0]]
                self.strategy_profile["deploy_order"] = order
        else:
            # Converge exploration slightly as reward improves
            sigma = round(max(0.012, sigma * 0.92), 4)
            if used_slots:
                self.strategy_profile["deploy_order"] = list(used_slots)

        self.strategy_profile["min_deploy_radius"] = min_r
        self.strategy_profile["explore_sigma"] = sigma
        self.strategy_profile["raids_self_learned"] = raids_learned
        self.strategy_profile["last_raid_reward"] = round(global_reward, 3)

        os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "samples": len(X_rows),
                "final_loss": final_loss,
                "strategy_profile": self.strategy_profile,
            },
            self.weights_path,
        )

        # Log self-learning summary to dataset/self_learned_raids.jsonl
        try:
            default_w = os.path.normcase(os.path.abspath(user_path(DEFAULT_POLICY_WEIGHTS)))
            curr_w = os.path.normcase(os.path.abspath(self.weights_path))
            self_learn_file = (
                user_path(DEFAULT_SELF_LEARN_PATH)
                if curr_w == default_w
                else os.path.join(os.path.dirname(self.weights_path), "self_learned_raids.jsonl")
            )
            os.makedirs(os.path.dirname(self_learn_file), exist_ok=True)
            with open(self_learn_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "timestamp": os.path.getmtime(self.weights_path),
                            "reward": round(global_reward, 4),
                            "is_mistake_correction": is_mistake,
                            "loot_ratio": round(loot_ratio, 4),
                            "drain_ratio": round(drain_val, 4),
                            "samples_updated": len(X_rows),
                            "loss_before": round(initial_loss, 5),
                            "loss_after": round(final_loss, 5),
                            "min_deploy_radius": min_r,
                            "explore_sigma": sigma,
                            "deploy_order": self.strategy_profile.get("deploy_order"),
                        }
                    )
                    + "\n"
                )
        except Exception as e:
            logger.debug("Could not append self-learning log: %s", e)

        self._episode_records = []
        return {
            "updated": True,
            "reward": round(global_reward, 3),
            "is_mistake": is_mistake,
            "samples": len(X_rows),
            "initial_loss": round(initial_loss, 5),
            "final_loss": round(final_loss, 5),
            "min_deploy_radius": min_r,
            "explore_sigma": sigma,
            "deploy_order": self.strategy_profile.get("deploy_order"),
        }


def _build_strategy_profile(records: list[dict]) -> dict:
    slot_counts: Counter = Counter()
    slot_first_seen: defaultdict = defaultdict(list)
    durations = []
    radii = []

    for rec in records:
        dur = float(rec.get("duration_s", 90.0) or 90.0)
        durations.append(dur)
        taps = rec.get("taps") or []
        seen_in_demo = {}
        for idx, tap in enumerate(taps):
            slot = str(tap.get("slot") or "s_goblin")
            slot_counts[slot] += 1
            if slot not in seen_in_demo:
                seen_in_demo[slot] = idx
            tx, ty = float(tap.get("x", 0.5)), float(tap.get("y", 0.5))
            r = math.hypot(tx - 0.50, ty - 0.45)
            if r >= 0.18:
                radii.append(r)
        for slot, first_idx in seen_in_demo.items():
            slot_first_seen[slot].append(first_idx)

    total_taps = max(1, sum(slot_counts.values()))
    slot_weights = {
        slot: round(cnt / total_taps, 4) for slot, cnt in slot_counts.items()
    }

    # Order slots by how early the player uses them on average, breaking ties by frequency
    all_slots = list(slot_counts.keys())
    all_slots.sort(
        key=lambda s: (
            sum(slot_first_seen[s]) / max(1, len(slot_first_seen[s])),
            -slot_counts[s],
        )
    )

    avg_duration = round(sum(durations) / len(durations), 1) if durations else 95.0
    p25_radius = float(np.percentile(radii, 25)) if radii else 0.26
    min_deploy_radius = round(max(0.22, min(0.32, p25_radius)), 3)

    return {
        "deploy_order": all_slots,
        "slot_weights": slot_weights,
        "surrender_after_s": max(45.0, min(150.0, avg_duration)),
        "min_deploy_radius": min_deploy_radius,
        "demos_count": len(records),
        "total_taps_learned": total_taps,
    }


def train_policy_from_dataset(
    dataset_path: str | None = None,
    weights_out: str | None = None,
    epochs: int = 80,
    lr: float = 3e-3,
) -> dict:
    """
    Train AttackPolicyNet + Strategic Profile via Behavioral Cloning on extracted video demonstrations.
    Runs on CUDA (RTX 4060) when available.
    """
    torch, nn = _get_torch()
    dataset_path = dataset_path or user_path(DEFAULT_DATASET_PATH)
    weights_out = weights_out or user_path(DEFAULT_POLICY_WEIGHTS)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    records = []
    X_rows = []
    Y_rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            taps = record.get("taps") or []
            for idx, tap in enumerate(taps):
                sx = float(tap.get("nearest_store_x", tap.get("x", 0.5)))
                sy = float(tap.get("nearest_store_y", tap.get("y", 0.5)))
                s_cls = str(tap.get("nearest_store_cls", "gold"))
                slot = str(tap.get("slot") or "s_goblin")
                tx = float(tap.get("x", sx))
                ty = float(tap.get("y", sy))
                # Skip interior noise/explosions too close to the center of the map
                if len(taps) > 10 and math.hypot(tx - 0.50, ty - 0.45) < 0.16:
                    continue
                dt = float(tap.get("dt", 0.08))
                dx = max(-0.22, min(0.22, tx - sx))
                dy = max(-0.22, min(0.22, ty - sy))
                X_rows.append(
                    PolicyPlanner.featurize_store(
                        sx, sy, s_cls, idx % 4, 4, troop_slot=slot
                    )
                )
                Y_rows.append([dx, dy, max(0.02, min(0.25, dt))])

    if not X_rows:
        raise ValueError("No valid tap samples found in dataset")

    strategy_profile = _build_strategy_profile(records)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    net = PolicyPlanner._build_network(torch, nn).to(device)
    net.train()

    X_tensor = torch.tensor(X_rows, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y_rows, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    initial_loss = None
    final_loss = 0.0
    for _epoch in range(max(1, int(epochs))):
        optimizer.zero_grad()
        pred = net(X_tensor)
        loss = loss_fn(pred, Y_tensor)
        if initial_loss is None:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(
        {
            "model_state": net.state_dict(),
            "samples": len(X_rows),
            "final_loss": final_loss,
            "strategy_profile": strategy_profile,
        },
        weights_out,
    )
    return {
        "device": device,
        "samples": len(X_rows),
        "initial_loss": round(initial_loss or 0.0, 5),
        "final_loss": round(final_loss, 5),
        "weights_path": weights_out,
        "strategy_profile": strategy_profile,
    }
