import json
import logging
import math
import os

import numpy as np

from core.utils import user_path

logger = logging.getLogger(__name__)

DEFAULT_POLICY_WEIGHTS = "models/policy_net.pt"
DEFAULT_DATASET_PATH = "dataset/attack_demos.jsonl"


def _get_torch():
    import torch
    import torch.nn as nn

    return torch, nn


class PolicyPlanner:
    """
    Deep Learning Behavioral Cloning Policy for Clash of Clans troop deployment.
    Learns human-like deployment offsets (dx, dy) relative to detected storages and
    base center geometry from recorded attack videos, executing on GPU (RTX 4060 CUDA).
    """

    def __init__(self, weights_path: str | None = None, device: str | None = None):
        self.weights_path = weights_path or user_path(DEFAULT_POLICY_WEIGHTS)
        self.device = device
        self.model = None
        self.loaded = False
        self.reload()

    @staticmethod
    def _build_network(torch, nn):
        class AttackPolicyNet(nn.Module):
            """
            Input features (8 dims):
              [store_x, store_y, vx_to_center, vy_to_center, dist_to_center,
               is_gold, is_elixir, ring_slot_norm]
            Output (3 dims):
              [offset_dx, offset_dy, inter_tap_delay_s]
              where offset_dx, offset_dy are normalized screen offsets in [-0.20, 0.20].
            """

            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(8, 64),
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
                dx = raw[:, 0:1] * 0.18
                dy = raw[:, 1:2] * 0.18
                delay = (raw[:, 2:3] + 1.0) * 0.12 + 0.02
                return torch.cat([dx, dy, delay], dim=1)

        return AttackPolicyNet()

    def reload(self) -> bool:
        if not os.path.isfile(self.weights_path):
            self.loaded = False
            return False
        try:
            torch, nn = _get_torch()
            if self.device is None:
                self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            net = self._build_network(torch, nn).to(self.device)
            state = torch.load(
                self.weights_path, map_location=self.device, weights_only=True
            )
            net.load_state_dict(state["model_state"])
            net.eval()
            self.model = net
            self.loaded = True
            logger.info(
                "Loaded Deep Learning AttackPolicyNet from %s on %s",
                self.weights_path,
                self.device,
            )
            return True
        except Exception as e:
            logger.warning("Could not load AttackPolicyNet weights: %s", e)
            self.loaded = False
            return False

    @staticmethod
    def featurize_store(
        sx: float, sy: float, cls_name: str, slot_k: int, max_k: int = 4
    ) -> list[float]:
        vx = 0.5 - sx
        vy = 0.46 - sy
        dist = math.hypot(vx, vy)
        is_gold = 1.0 if "gold" in str(cls_name).lower() else 0.0
        is_elixir = (
            1.0
            if "elixir" in str(cls_name).lower() and "d_" not in str(cls_name).lower()
            else 0.0
        )
        k_norm = float(slot_k) / max(1, max_k - 1) if max_k > 1 else 0.0
        return [sx, sy, vx, vy, dist, is_gold, is_elixir, k_norm]

    def predict_points(
        self,
        base: dict,
        width: int,
        height: int,
        side: str = "all",
        y_limit: int | None = None,
        surrender_zone_fn=None,
        per_store: int = 4,
        max_points: int = 64,
    ) -> list[tuple[int, int]]:
        """
        Use the trained neural network to predict human-like deployment coordinates
        around detected storages.
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
        for cls_key in ("golds", "elixirs", "d_elixirs"):
            for pt in base.get(cls_key, []):
                try:
                    cx, cy = float(pt[0]), float(pt[1])
                except (TypeError, IndexError, ValueError):
                    continue
                nx, ny = cx / w, cy / h
                if not (x_bounds[0] <= nx <= x_bounds[1]):
                    continue
                for k in range(per_store):
                    features.append(self.featurize_store(nx, ny, cls_key, k, per_store))
                    anchors.append((nx, ny, k))

        if not features:
            return []

        with torch.no_grad():
            tensor_in = torch.tensor(
                features, dtype=torch.float32, device=self.device
            )
            preds = self.model(tensor_in).cpu().numpy()

        points = []
        for (nx, ny, k), pred in zip(anchors, preds):
            dx, dy, _delay = float(pred[0]), float(pred[1]), float(pred[2])
            angle_spread = (k - (per_store - 1) / 2.0) * 0.35
            cos_a, sin_a = math.cos(angle_spread), math.sin(angle_spread)
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a

            px = int(round((nx + rx) * w))
            py = int(round((ny + ry) * h))
            px = max(15, min(int(w) - 100, px))
            py = max(15, min(limit - 5, py))
            if surrender_zone_fn is not None and surrender_zone_fn(px, py):
                continue
            if any((px - ox) ** 2 + (py - oy) ** 2 < 100 for ox, oy in points):
                continue
            points.append((px, py))
            if len(points) >= max_points:
                break

        return points


def train_policy_from_dataset(
    dataset_path: str | None = None,
    weights_out: str | None = None,
    epochs: int = 80,
    lr: float = 3e-3,
) -> dict:
    """
    Train AttackPolicyNet via Behavioral Cloning on extracted video demonstrations.
    Runs on CUDA (RTX 4060) when available.
    """
    torch, nn = _get_torch()
    dataset_path = dataset_path or user_path(DEFAULT_DATASET_PATH)
    weights_out = weights_out or user_path(DEFAULT_POLICY_WEIGHTS)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    X_rows = []
    Y_rows = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            taps = record.get("taps") or []
            for idx, tap in enumerate(taps):
                sx = float(tap.get("nearest_store_x", tap.get("x", 0.5)))
                sy = float(tap.get("nearest_store_y", tap.get("y", 0.5)))
                s_cls = str(tap.get("nearest_store_cls", "gold"))
                tx = float(tap.get("x", sx))
                ty = float(tap.get("y", sy))
                dt = float(tap.get("dt", 0.08))
                dx = max(-0.18, min(0.18, tx - sx))
                dy = max(-0.18, min(0.18, ty - sy))
                X_rows.append(
                    PolicyPlanner.featurize_store(sx, sy, s_cls, idx % 4, 4)
                )
                Y_rows.append([dx, dy, max(0.02, min(0.25, dt))])

    if not X_rows:
        raise ValueError("No valid tap samples found in dataset")

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
        },
        weights_out,
    )
    return {
        "samples": len(X_rows),
        "device": device,
        "initial_loss": round(initial_loss or 0.0, 6),
        "final_loss": round(final_loss, 6),
        "weights_path": weights_out,
    }
