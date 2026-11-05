import glob
import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from core.utils import user_path

logger = logging.getLogger(__name__)

DEFAULT_COMMANDER_WEIGHTS = "models/coc_commander.pt"
DEFAULT_SELF_LEARN_LOG = "dataset/self_learned_raids.jsonl"
SIDE_CHOICES = ("all", "left", "right")


def evaluate_loot_targets(
    loot: dict,
    targets: dict,
    mode: str = "and",
    dead_base: bool = False,
) -> bool:
    """
    Evaluate whether base loot satisfies target thresholds under:
    - 'and': all resources meet their respective targets
    - 'or': at least one resource with a positive target meets its target
    - 'smart': standard AND, OR combined Gold+Elixir >= 1.15x target sum,
      OR single jackpot >= 1.6x target, plus a 15% threshold discount on Dead Bases.
    """
    mode_norm = str(mode or "and").strip().lower()
    gold = int(loot.get("gold", 0) or 0)
    elixir = int(loot.get("elixir", 0) or 0)
    delixir = int(loot.get("delixir", 0) or 0)

    g_req = int(targets.get("gold", 0) or 0)
    e_req = int(targets.get("elixir", 0) or 0)
    d_req = int(targets.get("delixir", 0) or 0)

    if mode_norm == "or":
        checks = []
        if g_req > 0:
            checks.append(gold >= g_req)
        if e_req > 0:
            checks.append(elixir >= e_req)
        if d_req > 0:
            checks.append(delixir >= d_req)
        return any(checks) if checks else True

    if mode_norm == "smart":
        discount = 0.85 if dead_base else 1.0
        sg = int(g_req * discount)
        se = int(e_req * discount)
        sd = int(d_req * discount)
        if gold >= sg and elixir >= se and delixir >= sd:
            return True
        if (sg + se) > 0 and (gold + elixir) >= int((sg + se) * 1.15):
            if sd == 0 or delixir >= int(sd * 0.7):
                return True
        if sg > 0 and gold >= int(sg * 1.6) and (se == 0 or elixir >= int(se * 0.5)):
            return True
        if se > 0 and elixir >= int(se * 1.6) and (sg == 0 or gold >= int(sg * 0.5)):
            return True
        return False

    return gold >= g_req and elixir >= e_req and delixir >= d_req


def infer_target_side(base_geom: dict) -> str:
    """
    Choose 'left' or 'right' if >= 72% of detected stores cluster on one side, else 'all'.
    """
    if not isinstance(base_geom, dict):
        return "all"
    storages = base_geom.get("storages") or []
    if len(storages) < 4:
        return "all"
    left_count = sum(1 for s in storages if float(s.get("x", 0.5)) <= 0.48)
    right_count = sum(1 for s in storages if float(s.get("x", 0.5)) >= 0.52)
    total = len(storages)
    if left_count / total >= 0.72:
        return "left"
    if right_count / total >= 0.72:
        return "right"
    return "all"


def extract_base_features(state: dict) -> list[float]:
    """
    Encode a Clash of Clans base state into a 16-dimensional tactical feature vector
    specifically designed for CoCCommanderNet.
    """
    loot = state.get("loot") if isinstance(state.get("loot"), dict) else {}
    targets = state.get("targets") if isinstance(state.get("targets"), dict) else {}
    base_geom = state.get("base") if isinstance(state.get("base"), dict) else {}
    attack_cfg = state.get("attack") if isinstance(state.get("attack"), dict) else {}

    gold = float(loot.get("gold", 0) or 0)
    elixir = float(loot.get("elixir", 0) or 0)
    delixir = float(loot.get("delixir", 0) or 0)

    t_gold = float(targets.get("gold", 0) or 0)
    t_elixir = float(targets.get("elixir", 0) or 0)
    t_dark = float(targets.get("delixir", 0) or 0)

    gold_ratio = min(3.0, gold / max(100_000.0, t_gold)) if t_gold > 0 else min(3.0, gold / 600_000.0)
    elixir_ratio = min(3.0, elixir / max(100_000.0, t_elixir)) if t_elixir > 0 else min(3.0, elixir / 600_000.0)
    dark_ratio = min(3.0, delixir / max(1_000.0, t_dark)) if t_dark > 0 else min(2.0, delixir / 5_000.0)
    combo_ratio = min(
        3.0,
        (gold + elixir) / max(200_000.0, t_gold + t_elixir),
    )

    storages = base_geom.get("storages") if isinstance(base_geom.get("storages"), list) else []
    n_stores = len(storages)
    store_count_norm = min(1.0, n_stores / 16.0)

    if n_stores > 0:
        xs = [float(s.get("x", 0.5)) for s in storages]
        ys = [float(s.get("y", 0.46)) for s in storages]
        dists = [math.hypot(x - 0.50, y - 0.46) for x, y in zip(xs, ys)]
        avg_dist = float(sum(dists) / n_stores)
        outer_frac = sum(1 for d in dists if d >= 0.21) / float(n_stores)
        left_frac = sum(1 for x in xs if x <= 0.48) / float(n_stores)
        right_frac = sum(1 for x in xs if x >= 0.52) / float(n_stores)
        top_frac = sum(1 for y in ys if y <= 0.45) / float(n_stores)
        bottom_frac = sum(1 for y in ys if y > 0.45) / float(n_stores)
        spread_std = float(np.std(xs) + np.std(ys))
        gold_frac = sum(
            1 for s in storages if "gold" in str(s.get("type") or s.get("cls") or "").lower()
        ) / float(n_stores)
        elixir_frac = sum(
            1 for s in storages if "elixir" in str(s.get("type") or s.get("cls") or "").lower()
        ) / float(n_stores)
    else:
        avg_dist = float(base_geom.get("avg_center_dist", 0.20) or 0.20)
        outer_frac = 1.0 if avg_dist >= 0.22 else 0.3
        left_frac = 0.5
        right_frac = 0.5
        top_frac = 0.5
        bottom_frac = 0.5
        spread_std = 0.25
        gold_frac = 0.5
        elixir_frac = 0.5

    dead_base = bool(base_geom.get("dead_base", False)) or avg_dist >= 0.23
    target_mode = str(attack_cfg.get("target_mode") or "smart")
    meets_filter = evaluate_loot_targets(loot, targets, mode=target_mode, dead_base=dead_base)

    return [
        round(gold_ratio, 4),
        round(elixir_ratio, 4),
        round(dark_ratio, 4),
        round(combo_ratio, 4),
        1.0 if dead_base else 0.0,
        round(min(1.0, avg_dist * 2.5), 4),
        round(store_count_norm, 4),
        round(left_frac, 4),
        round(right_frac, 4),
        round(top_frac, 4),
        round(bottom_frac, 4),
        round(outer_frac, 4),
        round(min(1.0, spread_std * 2.0), 4),
        round(gold_frac, 4),
        round(elixir_frac, 4),
        1.0 if meets_filter else 0.0,
    ]


class CoCCommanderNet(nn.Module):
    """
    Dedicated Clash of Clans Multi-Head Strategic Neural Network.
    Replaces bulky general-purpose LLMs with a <1ms CUDA tactical brain specialized 100% for CoC:
      - Head 1 (profit_head): Predicts expected loot capture ratio [0..1]
      - Head 2 (side_head): Logits over attack directions ('all', 'left', 'right')
      - Head 3 (style_head): Logits over deployment doctrine (0 = learned combo/tanks first, 1 = fast goblin blitz first)
      - Head 4 (timer_head): Normalized optimal surrender timer [0..1] -> [45s..140s]
    """

    def __init__(self, in_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.res_block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.profit_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.side_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, len(SIDE_CHOICES)),
        )
        self.style_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 2),
        )
        self.timer_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h + self.res_block(h)
        profit = self.profit_head(h)
        side_logits = self.side_head(h)
        style_logits = self.style_head(h)
        timer_norm = self.timer_head(h)
        return profit, side_logits, style_logits, timer_norm


class Brain:
    name = "base"

    def plan(self, state: dict) -> dict:
        raise NotImplementedError


class RuleBrain(Brain):
    """Deterministic fallback strategy."""

    name = "rule"
    enabled = True

    def describe(self) -> str:
        return "rule-based tactical planner"

    def plan(self, state: dict) -> dict:
        try:
            loot = state["loot"]
            targets = state["targets"]
            attack_cfg = state.get("attack") if isinstance(state.get("attack"), dict) else {}
            base_geom = state.get("base") if isinstance(state.get("base"), dict) else {}
            target_mode = attack_cfg.get("target_mode", "smart")
            dead_base = bool(base_geom.get("dead_base", False))
            meets = evaluate_loot_targets(
                loot, targets, mode=target_mode, dead_base=dead_base
            )
            deploy_order = list(
                state.get("learned_deploy_order") or state.get("troop_slots") or ["s_goblin"]
            )
            surrender_s = float(
                state.get("learned_surrender_s")
                or attack_cfg.get("time_limit", 90.0)
            )
            return {
                "skip_base": not meets,
                "deploy_order": deploy_order,
                "surrender_after_s": surrender_s,
                "target_side": infer_target_side(base_geom),
            }
        except Exception as e:
            logger.debug(f"RuleBrain could not plan from state: {e}")
            return {}


class CoCTacticalBrain(Brain):
    """
    100% Dedicated Clash of Clans Deep Learning Strategic Brain.
    Runs CoCCommanderNet on GPU (CUDA) in <1ms with zero external APIs or LLM overhead,
    and continuously self-learns from every raid outcome.
    """

    name = "coc_ai"

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        self.weights_path = weights_path or user_path(DEFAULT_COMMANDER_WEIGHTS)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = CoCCommanderNet().to(self.device)
        self.enabled = True
        self.loaded = False
        self.last_error = ""
        self.stats = {
            "raids_learned": 0,
            "avg_reward": 0.65,
            "preferred_order": [],
        }
        self._load_or_bootstrap()

    def describe(self) -> str:
        return (
            f"CoCCommanderNet ({self.device}, "
            f"raids_learned={self.stats.get('raids_learned', 0)})"
        )

    def _load_or_bootstrap(self):
        if os.path.isfile(self.weights_path):
            try:
                ckpt = torch.load(
                    self.weights_path, map_location=self.device, weights_only=True
                )
                state_dict = ckpt.get("model_state", ckpt)
                self.model.load_state_dict(state_dict)
                if isinstance(ckpt.get("stats"), dict):
                    self.stats.update(ckpt["stats"])
                self.model.eval()
                self.loaded = True
                return
            except Exception as e:
                logger.warning(f"Rebuilding CoCCommanderNet weights ({e})")

        try:
            train_commander_from_data(weights_out=self.weights_path, device=self.device)
            if os.path.isfile(self.weights_path):
                ckpt = torch.load(
                    self.weights_path, map_location=self.device, weights_only=True
                )
                self.model.load_state_dict(ckpt.get("model_state", ckpt))
                if isinstance(ckpt.get("stats"), dict):
                    self.stats.update(ckpt["stats"])
                self.model.eval()
                self.loaded = True
        except Exception as e:
            logger.warning(f"CoCCommanderNet bootstrap failed: {e}")

    def reload(self) -> bool:
        self._load_or_bootstrap()
        return self.loaded

    def _order_for_style(self, slots: list[str], learned_order: list[str], style_idx: int) -> list[str]:
        valid_learned = [s for s in learned_order if s in slots]
        for s in slots:
            if s not in valid_learned:
                valid_learned.append(s)
        if not valid_learned:
            return list(slots)

        if style_idx == 1 and "s_goblin" in valid_learned:
            # Style 1: Fast Sneaky Goblin collector blitz first, then support troops
            return ["s_goblin"] + [s for s in valid_learned if s != "s_goblin"]

        # Style 0: Learned video sequence (e.g. Wall Breakers / Tanks / Main slots -> Goblins)
        return valid_learned

    def plan(self, state: dict) -> dict:
        try:
            feats = extract_base_features(state)
            x_t = torch.tensor([feats], dtype=torch.float32, device=self.device)
            self.model.eval()
            with torch.no_grad():
                profit_t, side_logits, style_logits, timer_norm = self.model(x_t)

            pred_profit = float(profit_t[0, 0].item())
            side_idx = int(torch.argmax(side_logits[0]).item())
            style_idx = int(torch.argmax(style_logits[0]).item())
            raw_timer = 45.0 + float(timer_norm[0, 0].item()) * 95.0

            loot = state.get("loot") or {}
            targets = state.get("targets") or {}
            attack_cfg = state.get("attack") if isinstance(state.get("attack"), dict) else {}
            base_geom = state.get("base") if isinstance(state.get("base"), dict) else {}
            dead_base = bool(base_geom.get("dead_base", False))
            target_mode = str(attack_cfg.get("target_mode") or "smart")

            meets_targets = evaluate_loot_targets(
                loot, targets, mode=target_mode, dead_base=dead_base
            )
            # CoCCommanderNet attacks whenever loot filter passes OR when neural net predicts
            # high stealable profit (>= 0.72) on a dead/exposed base with decent resources.
            combo_ratio = feats[3]
            if meets_targets:
                skip_base = False
            elif dead_base and pred_profit >= 0.72 and combo_ratio >= 0.80:
                skip_base = False
            else:
                skip_base = True

            target_side = SIDE_CHOICES[side_idx] if 0 <= side_idx < len(SIDE_CHOICES) else "all"
            # Guard against picking a single flank when storages are evenly spread across both sides
            left_frac, right_frac = feats[7], feats[8]
            if target_side == "left" and left_frac < 0.65:
                target_side = "all"
            elif target_side == "right" and right_frac < 0.65:
                target_side = "all"

            slots = list(state.get("troop_slots") or ["s_goblin"])
            learned_order = list(
                state.get("learned_deploy_order")
                or self.stats.get("preferred_order")
                or slots
            )
            deploy_order = self._order_for_style(slots, learned_order, style_idx)

            learned_surrender = float(
                state.get("learned_surrender_s")
                or attack_cfg.get("time_limit", 90.0)
            )
            surrender_s = round(
                max(30.0, min(160.0, 0.55 * raw_timer + 0.45 * learned_surrender)),
                1,
            )

            self.last_error = ""
            return {
                "skip_base": skip_base,
                "deploy_order": deploy_order,
                "surrender_after_s": surrender_s,
                "target_side": target_side,
                "predicted_profit": round(pred_profit, 3),
                "style_idx": style_idx,
            }
        except Exception as e:
            self.last_error = str(e)
            logger.warning(f"CoCTacticalBrain plan error: {e}")
            return RuleBrain().plan(state)

    def learn_from_raid(
        self,
        state: dict,
        plan_used: dict,
        loot_gain: dict,
        base_loot: dict,
        drained: float | None,
        lr: float = 2e-3,
        steps: int = 8,
    ) -> dict:
        """
        Online Reinforcement Learning for the Macro Strategic Brain (CoCCommanderNet).
        Updates profit prediction, attack side selection, troop deployment doctrine,
        and surrender timer directly from the completed raid's outcome.
        """
        try:
            feats = extract_base_features(state)
            g_base = max(1, int(base_loot.get("gold", 0) or 0))
            e_base = max(1, int(base_loot.get("elixir", 0) or 0))
            g_gain = max(0, int(loot_gain.get("gold", 0) or 0))
            e_gain = max(0, int(loot_gain.get("elixir", 0) or 0))

            loot_ratio = min(1.0, (g_gain + e_gain) / float(g_base + e_base))
            drain_val = float(drained) if drained is not None else loot_ratio
            realized_reward = 0.60 * loot_ratio + 0.40 * drain_val

            prev_avg = float(self.stats.get("avg_reward", 0.60))
            advantage = realized_reward - prev_avg
            is_mistake = realized_reward < 0.48 or advantage < -0.10

            used_side = str(plan_used.get("target_side") or "all")
            used_side_idx = SIDE_CHOICES.index(used_side) if used_side in SIDE_CHOICES else 0
            used_style_idx = int(plan_used.get("style_idx", 0))

            if not is_mistake:
                target_side_idx = used_side_idx
                target_style_idx = used_style_idx
            else:
                # Mistake correction: if single-side failed, switch to 'all' (or cluster side);
                # also flip troop deployment style (blitz <-> combo)
                inferred = infer_target_side(state.get("base") or {})
                if used_side != "all":
                    target_side_idx = 0
                elif inferred in SIDE_CHOICES and inferred != "all":
                    target_side_idx = SIDE_CHOICES.index(inferred)
                else:
                    target_side_idx = 0
                target_style_idx = 1 - used_style_idx

            # Optimal timer target: slightly faster if we drained 100%, longer if storages survived
            used_timer = float(plan_used.get("surrender_after_s", 90.0))
            if drain_val >= 0.90:
                opt_timer = max(45.0, used_timer * 0.95)
            elif is_mistake:
                opt_timer = min(140.0, used_timer * 1.12)
            else:
                opt_timer = used_timer
            target_timer_norm = max(0.0, min(1.0, (opt_timer - 45.0) / 95.0))

            x_t = torch.tensor([feats], dtype=torch.float32, device=self.device)
            y_profit = torch.tensor([[realized_reward]], dtype=torch.float32, device=self.device)
            y_side = torch.tensor([target_side_idx], dtype=torch.long, device=self.device)
            y_style = torch.tensor([target_style_idx], dtype=torch.long, device=self.device)
            y_timer = torch.tensor([[target_timer_norm]], dtype=torch.float32, device=self.device)

            self.model.train()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            loss_val = 0.0
            for _ in range(max(1, int(steps))):
                optimizer.zero_grad()
                p_pred, side_logits, style_logits, t_pred = self.model(x_t)
                loss_p = nn.functional.smooth_l1_loss(p_pred, y_profit)
                loss_side = nn.functional.cross_entropy(side_logits, y_side)
                loss_style = nn.functional.cross_entropy(style_logits, y_style)
                loss_t = nn.functional.smooth_l1_loss(t_pred, y_timer)
                loss = loss_p + 0.5 * loss_side + 0.5 * loss_style + 0.25 * loss_t
                loss.backward()
                optimizer.step()
                loss_val = float(loss.item())
            self.model.eval()

            self.stats["raids_learned"] = int(self.stats.get("raids_learned", 0)) + 1
            self.stats["avg_reward"] = round(0.82 * prev_avg + 0.18 * realized_reward, 4)
            if not is_mistake and plan_used.get("deploy_order"):
                self.stats["preferred_order"] = list(plan_used["deploy_order"])

            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            torch.save(
                {
                    "model_state": self.model.state_dict(),
                    "stats": self.stats,
                },
                self.weights_path,
            )
            return {
                "updated": True,
                "is_mistake": is_mistake,
                "reward": round(realized_reward, 3),
                "loss": round(loss_val, 5),
                "raids_learned": self.stats["raids_learned"],
            }
        except Exception as e:
            logger.warning(f"CoCTacticalBrain online learning failed: {e}")
            return {"updated": False, "reason": str(e)}


def train_commander_from_data(
    dataset_path: str | None = None,
    replay_dir: str | None = None,
    weights_out: str | None = None,
    epochs: int = 80,
    lr: float = 3e-3,
    device: str | None = None,
) -> dict:
    """
    Train CoCCommanderNet from:
      1. Video demonstrations in dataset/attack_demos.jsonl
      2. Historical raid logs in logs/raid_*.json
      3. Clash of Clans tactical domain priors (dead base blitz, flank clustering, core diving)
    """
    dataset_path = dataset_path or user_path("dataset/attack_demos.jsonl")
    replay_dir = replay_dir or user_path("logs")
    weights_out = weights_out or user_path(DEFAULT_COMMANDER_WEIGHTS)
    dev = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    X_rows = []
    Y_profit = []
    Y_side = []
    Y_style = []
    Y_timer = []
    preferred_order = []

    # 1. Load human video demonstrations
    if os.path.isfile(dataset_path):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    demo = json.loads(line)
                except Exception:
                    continue
                storages = demo.get("storages") or []
                order = demo.get("deploy_order") or []
                if order:
                    preferred_order = list(order)
                duration_s = float(demo.get("duration_s", 110.0) or 110.0)
                side = str(demo.get("target_side") or "all")
                side_idx = SIDE_CHOICES.index(side) if side in SIDE_CHOICES else 0
                style_idx = 1 if (order and order[0] == "s_goblin") else 0
                timer_norm = max(0.0, min(1.0, (duration_s - 45.0) / 95.0))

                state = {
                    "loot": {"gold": 850_000, "elixir": 850_000, "delixir": 6_000},
                    "targets": {"gold": 500_000, "elixir": 500_000, "delixir": 2_000},
                    "base": {"storages": storages, "dead_base": True},
                    "attack": {"target_mode": "smart"},
                }
                X_rows.append(extract_base_features(state))
                Y_profit.append([0.92])
                Y_side.append(side_idx)
                Y_style.append(style_idx)
                Y_timer.append([timer_norm])

    # 2. Load historical raid outcomes from logs/raid_*.json
    if os.path.isdir(replay_dir):
        for rpath in glob.glob(os.path.join(replay_dir, "raid_*.json"))[-200:]:
            try:
                with open(rpath, "r", encoding="utf-8") as f:
                    raid = json.load(f)
                state = raid.get("state") or {}
                outcome = raid.get("outcome") or {}
                if outcome.get("status") != "attacked":
                    continue
                base_loot = state.get("loot") or {}
                gain = outcome.get("loot_gain") or {}
                drained = outcome.get("drained")
                g_b = max(1, int(base_loot.get("gold", 0) or 0))
                e_b = max(1, int(base_loot.get("elixir", 0) or 0))
                g_g = max(0, int(gain.get("gold", 0) or 0))
                e_g = max(0, int(gain.get("elixir", 0) or 0))
                l_ratio = min(1.0, (g_g + e_g) / float(g_b + e_b))
                d_val = float(drained) if isinstance(drained, (int, float)) else l_ratio
                reward = 0.60 * l_ratio + 0.40 * d_val

                inferred_side = infer_target_side(state.get("base") or {})
                s_idx = SIDE_CHOICES.index(inferred_side) if inferred_side in SIDE_CHOICES else 0
                X_rows.append(extract_base_features(state))
                Y_profit.append([reward])
                Y_side.append(s_idx)
                Y_style.append(0)
                Y_timer.append([0.65])
            except Exception:
                continue

    # 3. Add CoC domain tactical priors so all 3 sides & styles are well-calibrated
    priors = [
        # (feats, profit, side_idx, style_idx, timer_norm)
        # All-around dead base with high outer loot -> surround ('all'), high profit
        ([1.6, 1.6, 1.2, 1.6, 1.0, 0.65, 0.85, 0.50, 0.50, 0.50, 0.50, 0.80, 0.60, 0.5, 0.5, 1.0], 0.92, 0, 0, 0.70),
        # Left-clustered storages -> attack 'left' (idx 1)
        ([1.3, 1.2, 1.0, 1.25, 0.0, 0.50, 0.65, 0.82, 0.18, 0.50, 0.50, 0.55, 0.35, 0.5, 0.5, 1.0], 0.82, 1, 0, 0.60),
        # Right-clustered storages -> attack 'right' (idx 2)
        ([1.2, 1.4, 1.0, 1.30, 0.0, 0.50, 0.65, 0.15, 0.85, 0.50, 0.50, 0.55, 0.35, 0.5, 0.5, 1.0], 0.82, 2, 0, 0.60),
        # Exposed collector ring -> fast blitz (style 1)
        ([1.4, 1.4, 0.8, 1.4, 1.0, 0.72, 0.90, 0.48, 0.52, 0.48, 0.52, 0.90, 0.65, 0.5, 0.5, 1.0], 0.95, 0, 1, 0.55),
        # Low-loot fortified core base -> low profit
        ([0.3, 0.3, 0.2, 0.3, 0.0, 0.30, 0.50, 0.50, 0.50, 0.50, 0.50, 0.15, 0.25, 0.5, 0.5, 0.0], 0.25, 0, 0, 0.40),
    ]
    for f_vec, prof, s_i, st_i, tm_n in priors:
        for _ in range(4):
            X_rows.append(f_vec)
            Y_profit.append([prof])
            Y_side.append(s_i)
            Y_style.append(st_i)
            Y_timer.append([tm_n])

    model = CoCCommanderNet().to(dev)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_t = torch.tensor(X_rows, dtype=torch.float32, device=dev)
    Yp_t = torch.tensor(Y_profit, dtype=torch.float32, device=dev)
    Ys_t = torch.tensor(Y_side, dtype=torch.long, device=dev)
    Yst_t = torch.tensor(Y_style, dtype=torch.long, device=dev)
    Yt_t = torch.tensor(Y_timer, dtype=torch.float32, device=dev)

    initial_loss = 0.0
    final_loss = 0.0
    for ep in range(max(1, int(epochs))):
        optimizer.zero_grad()
        p_pred, s_logits, st_logits, t_pred = model(X_t)
        loss = (
            nn.functional.smooth_l1_loss(p_pred, Yp_t)
            + 0.5 * nn.functional.cross_entropy(s_logits, Ys_t)
            + 0.4 * nn.functional.cross_entropy(st_logits, Yst_t)
            + 0.25 * nn.functional.smooth_l1_loss(t_pred, Yt_t)
        )
        if ep == 0:
            initial_loss = float(loss.item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    stats = {
        "raids_learned": len(X_rows),
        "avg_reward": 0.72,
        "preferred_order": preferred_order,
    }
    os.makedirs(os.path.dirname(weights_out), exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "stats": stats,
            "initial_loss": round(initial_loss, 5),
            "final_loss": round(final_loss, 5),
        },
        weights_out,
    )
    return {
        "samples": len(X_rows),
        "initial_loss": round(initial_loss, 5),
        "final_loss": round(final_loss, 5),
        "device": dev,
        "weights_path": weights_out,
    }
