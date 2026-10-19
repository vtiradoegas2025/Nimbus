#!/usr/bin/env python3
"""Qualify calm A-grid sensitivity to the acoustic-substep count.

Each input must be a strict 20-second hydrostatic-hold artifact produced from
the same initial state.  The checker first applies the ordinary hold gate, then
compares the physical-cell final states from three, six, and twelve acoustic
substeps.  This is a bounded-sensitivity gate for the production integrator;
it is not an acoustic-wave dispersion or timestep-convergence study.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


COMMON_TOOLS = Path(__file__).resolve().parents[2] / "common"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from check_hydrostatic_hold import (
    Float32Array,
    Frame,
    QualificationError,
    STEP_DIR_PATTERN,
    load_fields,
    parse_frame,
    physical_indices,
    qualify_run,
    read_json_object,
    require,
    valid_generation,
)


@dataclass(frozen=True)
class ComparisonThresholds:
    u_ms: float = 5.0e-5
    v_ms: float = 5.0e-5
    w_ms: float = 1.0e-4
    rho_kgm3: float = 2.0e-6
    p_pa: float = 5.0e-2
    theta_k: float = 2.0e-4
    qv_kgkg: float = 2.0e-8


FIELD_LIMIT_NAMES = {
    "u": "u_ms",
    "v": "v_ms",
    "w": "w_ms",
    "rho": "rho_kgm3",
    "p": "p_pa",
    "theta": "theta_k",
    "qv": "qv_kgkg",
}


def load_initial_final(run_dir: Path) -> tuple[Frame, Mapping[str, Float32Array], Frame, Mapping[str, Float32Array]]:
    marker_path = run_dir / "nimbus_output.json"
    marker = read_json_object(marker_path)
    generation = marker.get("publication_generation")
    require(valid_generation(generation), f"{marker_path}: invalid publication generation")
    assert isinstance(generation, str)
    paths = sorted(
        path for path in run_dir.iterdir()
        if path.is_dir() and STEP_DIR_PATTERN.fullmatch(path.name)
    )
    require(len(paths) == 2, f"{run_dir}: expected exactly two frames")
    frames = sorted(
        (parse_frame(path, generation) for path in paths),
        key=lambda frame: frame.time_s,
    )
    return frames[0], load_fields(frames[0]), frames[1], load_fields(frames[1])


def max_abs_difference(
    lhs: Float32Array,
    rhs: Float32Array,
    frame: Frame,
) -> float:
    return max(
        abs(lhs.at(i, j, k) - rhs.at(i, j, k))
        for i, j, k in physical_indices(frame)
    )


def qualify_sweep(
    run_dirs: Mapping[int, Path],
    thresholds: ComparisonThresholds = ComparisonThresholds(),
) -> dict[str, object]:
    require(set(run_dirs) == {3, 6, 12}, "sweep must contain 3, 6, and 12 substeps")
    hold_results = {
        count: qualify_run(path, minimum_duration_s=20.0)
        for count, path in run_dirs.items()
    }
    loaded = {count: load_initial_final(path) for count, path in run_dirs.items()}

    reference_initial_frame, reference_initial, reference_final_frame, reference_final = loaded[12]
    for count in (3, 6):
        initial_frame, initial, final_frame, _ = loaded[count]
        require(initial_frame.shape == reference_initial_frame.shape, f"{count}: initial shape differs")
        require(final_frame.shape == reference_final_frame.shape, f"{count}: final shape differs")
        require(
            initial_frame.valid_lower == reference_initial_frame.valid_lower
            and initial_frame.valid_upper == reference_initial_frame.valid_upper,
            f"{count}: physical index range differs",
        )
        for field in FIELD_LIMIT_NAMES:
            difference = max_abs_difference(initial[field], reference_initial[field], initial_frame)
            require(difference == 0.0, f"{count}: initial {field} differs from 12-substep run")

    comparisons: dict[str, dict[str, float]] = {}
    for count in (3, 6):
        _, _, frame, final = loaded[count]
        metrics: dict[str, float] = {}
        for field, limit_name in FIELD_LIMIT_NAMES.items():
            difference = max_abs_difference(final[field], reference_final[field], frame)
            limit = getattr(thresholds, limit_name)
            require(math.isfinite(difference), f"{count}-vs-12 {field}: non-finite difference")
            require(
                difference <= limit,
                f"{count}-vs-12 {field} max difference {difference:.9g} exceeds {limit:.9g}",
            )
            metrics[f"max_abs_{field}"] = difference
        comparisons[f"{count}_vs_12"] = metrics

    return {
        "status": "PASS",
        "scope": "calm_hydrostatic_production_integrator",
        "substep_counts": [3, 6, 12],
        "hold_metrics": {
            str(count): hold_results[count]["metrics"] for count in (3, 6, 12)
        },
        "final_state_comparisons": comparisons,
        "comparison_thresholds": asdict(thresholds),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_3", type=Path)
    parser.add_argument("run_6", type=Path)
    parser.add_argument("run_12", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        result = qualify_sweep({3: args.run_3, 6: args.run_6, 12: args.run_12})
    except QualificationError as error:
        parser.exit(1, f"[agrid-acoustic-sweep] FAIL: {error}\n")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
