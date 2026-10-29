import argparse
import json
import sys
import time

from brain import DEFAULT_MODEL, GeminiBrain, RuleBrain
from settings import load_settings


def sample_state(settings):
    troops = settings.get("troops") or {}
    attack = settings.get("attack") or {}
    slots = troops.get("slots") or ["s_goblin"]
    if not isinstance(slots, (list, tuple)):
        slots = [slots]
    return {
        "loot": {"gold": 450000, "elixir": 520000, "delixir": 2500},
        "targets": {"gold": 400000, "elixir": 400000, "delixir": 0},
        "providers": {"golds": 3, "elixirs": 2, "d_elixirs": 1},
        "base": {
            "storages": [
                {"cls": "gold", "x": 0.31, "y": 0.28},
                {"cls": "gold", "x": 0.44, "y": 0.35},
                {"cls": "elixir", "x": 0.62, "y": 0.3},
                {"cls": "elixir", "x": 0.58, "y": 0.47},
                {"cls": "delixir", "x": 0.5, "y": 0.22},
            ],
            "bbox": {"x1": 0.31, "y1": 0.22, "x2": 0.62, "y2": 0.47},
        },
        "history": [
            {
                "gold": 320000,
                "elixir": 410000,
                "delixir": 1200,
                "drained": 0.72,
                "base_loot": {"gold": 500000, "elixir": 600000, "delixir": 2000},
            }
        ],
        "session": {
            "raids": 1,
            "totals": {"gold": 320000, "elixir": 410000, "delixir": 1200},
        },
        "troop_slots": list(slots),
        "deploy_mode": troops.get("deploy_mode", "tap"),
        "attack": {
            "drain_ratio": float(attack.get("drain_ratio", 0.8)),
            "time_limit": float(attack.get("time_limit", 90.0)),
        },
    }


def load_state(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("state"), dict):
        return data["state"]
    return data


def gemini_from_settings(settings):
    cfg = settings.get("brain") or {}
    return GeminiBrain(
        model=cfg.get("model", DEFAULT_MODEL),
        api_key=cfg.get("api_key") or None,
        api_key_env=cfg.get("api_key_env", "GEMINI_API_KEY"),
        timeout_s=float(cfg.get("timeout_s", 8.0)),
        endpoint=cfg.get("endpoint") or None,
        thinking_level=cfg.get("thinking_level", "low"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe rule and Gemini brains.")
    parser.add_argument(
        "--live", action="store_true", help="also call GeminiBrain over HTTP"
    )
    parser.add_argument(
        "--state", metavar="PATH", help="use a saved replay state instead of sample"
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    state = load_state(args.state) if args.state else sample_state(settings)

    print("state:")
    print(json.dumps(state, indent=2, ensure_ascii=False))
    print("rule plan:")
    print(json.dumps(RuleBrain().plan(state), indent=2, ensure_ascii=False))

    if not args.live:
        return 0

    brain = gemini_from_settings(settings)
    started = time.time()
    plan = brain.plan(state)
    elapsed = time.time() - started
    print(f"gemini plan ({brain.describe()}) in {elapsed:.2f}s:")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0 if plan else 1


if __name__ == "__main__":
    sys.exit(main())
