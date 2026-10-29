import argparse
import glob
import json
import os
import sys

PLAN_KEYS = ("skip_base", "deploy_order", "surrender_after_s", "target_side")


def load_raids(directory):
    pattern = os.path.join(directory, "raid_*.json")
    raids = []
    for path in glob.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"skip {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        data["_file"] = os.path.basename(path)
        raids.append(data)
    raids.sort(key=lambda r: r.get("_file", ""))
    return raids


def alt_plan_of(raid):
    for event in raid.get("events") or []:
        if isinstance(event, dict) and event.get("event") == "plan_alt":
            plan = event.get("plan")
            return event.get("name") or "alt", plan if isinstance(plan, dict) else {}
    return None, {}


def alt_error(raid):
    for event in raid.get("events") or []:
        if isinstance(event, dict) and event.get("event") == "plan_alt":
            return str(event.get("error") or "")
    return ""


def agreement(rule_plan, alt_plan):
    if not rule_plan or not alt_plan:
        return {}
    flags = {}
    for key in PLAN_KEYS:
        if key not in rule_plan or key not in alt_plan:
            continue
        rule_value = rule_plan[key]
        alt_value = alt_plan[key]
        if key == "deploy_order":
            flags[key] = list(rule_value or []) == list(alt_value or [])
        elif key == "surrender_after_s":
            try:
                flags[key] = abs(float(rule_value) - float(alt_value)) <= 10.0
            except (TypeError, ValueError):
                flags[key] = False
        else:
            flags[key] = rule_value == alt_value
    return flags


def summarise(raids):
    stats = {
        "raids": len(raids),
        "attacked": 0,
        "skipped": 0,
        "gold": 0,
        "elixir": 0,
        "delixir": 0,
        "drains": [],
        "alt_calls": 0,
        "alt_answers": 0,
        "alt_errors": {},
        "agree": {key: [0, 0] for key in PLAN_KEYS},
    }
    for raid in raids:
        outcome = raid.get("outcome") or {}
        status = outcome.get("status")
        if status == "attacked":
            stats["attacked"] += 1
            gain = outcome.get("loot_gain") or {}
            stats["gold"] += int(gain.get("gold", 0) or 0)
            stats["elixir"] += int(gain.get("elixir", 0) or 0)
            stats["delixir"] += int(gain.get("delixir", 0) or 0)
            drained = outcome.get("drained")
            if isinstance(drained, (int, float)):
                stats["drains"].append(float(drained))
        elif status == "skipped":
            stats["skipped"] += 1

        _, alt_plan = alt_plan_of(raid)
        stats["alt_calls"] += 1
        if alt_plan:
            stats["alt_answers"] += 1
        else:
            reason = alt_error(raid) or "unrecorded"
            stats["alt_errors"][reason] = stats["alt_errors"].get(reason, 0) + 1
        for key, matched in agreement(raid.get("plan") or {}, alt_plan).items():
            stats["agree"][key][0] += 1 if matched else 0
            stats["agree"][key][1] += 1
    return stats


def print_raids(raids):
    print(f"{'#':>3}  {'status':<9} {'gold':>9} {'elixir':>9} {'dark':>7} "
          f"{'drain':>6}  alt")
    for index, raid in enumerate(raids, 1):
        outcome = raid.get("outcome") or {}
        status = str(outcome.get("status") or "pending")
        gain = outcome.get("loot_gain") or {}
        drained = outcome.get("drained")
        drain_text = f"{float(drained) * 100:.0f}%" if isinstance(drained, (int, float)) else "-"
        name, alt_plan = alt_plan_of(raid)
        if not alt_plan:
            reason = alt_error(raid)
            alt_text = f"no answer ({reason})" if name and reason else (
                "no answer" if name else "-"
            )
        else:
            flags = agreement(raid.get("plan") or {}, alt_plan)
            diff = [key for key, matched in flags.items() if not matched]
            alt_text = f"{name}: " + ("match" if not diff else "diff " + ",".join(diff))
        print(
            f"{index:>3}  {status:<9} "
            f"{int(gain.get('gold', 0) or 0):>9} "
            f"{int(gain.get('elixir', 0) or 0):>9} "
            f"{int(gain.get('delixir', 0) or 0):>7} "
            f"{drain_text:>6}  {alt_text}"
        )


def print_summary(stats):
    print()
    print(f"raids: {stats['raids']}  attacked: {stats['attacked']}  "
          f"skipped: {stats['skipped']}")
    print(f"loot: Gold {stats['gold']:,} | Elixir {stats['elixir']:,} | "
          f"Dark {stats['delixir']:,}")
    if stats["drains"]:
        avg = sum(stats["drains"]) / len(stats["drains"])
        print(f"avg drain: {avg * 100:.0f}% over {len(stats['drains'])} raids")
    if stats["alt_calls"]:
        rate = stats["alt_answers"] / stats["alt_calls"] * 100
        print(f"alt plans: {stats['alt_answers']}/{stats['alt_calls']} "
              f"answered ({rate:.0f}%)")
        for reason, count in sorted(
            stats["alt_errors"].items(), key=lambda kv: -kv[1]
        ):
            print(f"  unanswered {count}x: {reason}")
        for key, (matches, total) in stats["agree"].items():
            if total:
                print(f"  agree {key}: {matches}/{total} "
                      f"({matches / total * 100:.0f}%)")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarise raid replay logs.")
    parser.add_argument("--dir", default="logs", help="replay directory (default: logs)")
    parser.add_argument("--last", type=int, default=0, help="only the last N raids")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.dir):
        print(f"no replay directory: {args.dir}", file=sys.stderr)
        return 1

    raids = load_raids(args.dir)
    if not raids:
        print(f"no raid files in {args.dir}", file=sys.stderr)
        return 1
    if args.last and args.last > 0:
        raids = raids[-args.last:]

    stats = summarise(raids)
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    print_raids(raids)
    print_summary(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
