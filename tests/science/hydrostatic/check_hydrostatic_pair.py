#!/usr/bin/env python3
"""Require matched dry/moist hydrostatic qualification results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence


def read_result(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "PASS":
        raise ValueError(f"{path} is not a passing hydrostatic result")
    return value


def check_pair(dry: dict[str, Any], moist: dict[str, Any]) -> dict[str, Any]:
    if dry.get("grid_shape") != moist.get("grid_shape"):
        raise ValueError("dry/moist grid shapes differ")
    if dry.get("duration_s") != moist.get("duration_s"):
        raise ValueError("dry/moist durations differ")
    try:
        dry_water = float(dry["inventories"]["initial"]["water"])
        moist_water = float(moist["inventories"]["initial"]["water"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("dry/moist results lack initial water inventories") from exc
    if not math.isfinite(dry_water) or not math.isfinite(moist_water):
        raise ValueError("dry/moist water inventories are non-finite")
    if dry_water < 0.0 or moist_water <= 0.0:
        raise ValueError("dry/moist water inventories have invalid signs")
    if dry_water >= 0.01 * moist_water:
        raise ValueError(
            "dry initial water inventory is not at least 100x below moist"
        )
    return {
        "schema": "nimbus.hydrostatic-pair.v1",
        "status": "PASS",
        "grid_shape": dry["grid_shape"],
        "duration_s": dry["duration_s"],
        "dry_initial_water": dry_water,
        "moist_initial_water": moist_water,
        "moist_to_dry_water_ratio": moist_water / max(dry_water, 1.0e-300),
        "dry_metrics": dry["metrics"],
        "moist_metrics": moist["metrics"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dry", type=Path)
    parser.add_argument("moist", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = check_pair(read_result(args.dry), read_result(args.moist))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.exit(1, f"[hydrostatic-pair] FAIL: {exc}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
